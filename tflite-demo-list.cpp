#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/version.h"
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

using namespace std;
using namespace cv;

#define KDRED "\u001b[31m"
#define KBLUH "\u001b[38;5;6m"
#define KNRM  "\033[0m"
#define KPURP "\033[35m"
#define KRED  "\033[31m"
#define KGRN  "\033[32m"
#define KGRND "\033[36m"
#define KYEL  "\033[33m"
#define KBLU  "\033[34m"

static void print_progressbar(unsigned int barLen, unsigned int curPos, unsigned int total, char *desc)
{
    printf("\r");
    int offset  = curPos*barLen/total;
    for(int i = 0; i < offset; i++)
        printf(KGRN "%s" KNRM, "■");
    for (uint32_t i = offset; i < barLen; i++)
        printf("■");
    printf(" [%6.2f%%] %03d/%03d " KGRN "[%s]" KNRM, curPos*100.0/total, curPos, total, desc);
    fflush(stdout);
    if (curPos == total)
        printf("\n");
}

static void showResult(float *pOut, uint32_t out_size)
{
    for(int i = 0 ; i < out_size; i++)
    {
        if ((0 != i)&& (0 == i % 16))
            printf("\n");
        printf("%10.6f, ", pOut[i]);
    }
    printf("\n");
}

static void from_y_normal(unsigned char* pY, int w, int h, float* dst, const float mean, const float scale, unsigned num_threads)
{
    int size = w * h;
    int nn = size >> 3;
    int i = 0;
    int remain = size & 7;

    float32x4_t mean32x4  = vdupq_n_f32(mean);
    float32x4_t scale32x4 = vdupq_n_f32(scale);
    float* ptr0 = dst;

    #pragma omp parallel for num_threads(num_threads)
    for ( i = 0; i < nn; i++)
    {
        float *pdst = ptr0 + 8*i;

        uint8x8_t _y = vld1_u8(pY + 8*i);
        uint16x8_t _y16  = vmovl_u8(_y);

        float32x4_t _ylow  = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_y16)));
        float32x4_t _yhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_y16)));

        _ylow  = vsubq_f32(_ylow, mean32x4);
        _yhigh = vsubq_f32(_yhigh, mean32x4);

        _ylow  = vmulq_f32(_ylow, scale32x4);
        _yhigh = vmulq_f32(_yhigh, scale32x4);

        vst1q_f32(pdst,   _ylow);
        vst1q_f32(pdst+4, _yhigh);
    }

    pY   += 8*nn;
    ptr0 += 8*nn;

    for (i = 0; i < remain; i++)
        *ptr0++ = ((float)*pY++ - mean)*scale;

    return;
}

static int makeDir(const char* inpath)
{
    int beginCmpPath;
    int endCmpPath;
    int fullPathLen;
    char path[512];
    int pathLen;
    char currentPath[128] = {0};
    char fullPath[128] = {0};

    strcpy(path, inpath);
    char *pSplit = strrchr(path, '.');
    if (NULL != pSplit)
    {
        pSplit = strrchr(path, '/');
        if (NULL != pSplit)
            *pSplit = 0;
    }
    pathLen = strlen(path);

    if ('/' != path[0])
    {
        getcwd(currentPath, sizeof(currentPath));
        strcat(currentPath, "/");
        beginCmpPath = strlen(currentPath);
        strcat(currentPath, path);
        if (path[pathLen] != '/')
            strcat(currentPath, "/");
        endCmpPath = strlen(currentPath);
    }
    else
    {
        int pathLen = strlen(path);
        strcpy(currentPath, path);
        if (path[pathLen] != '/')
            strcat(currentPath, "/");
        beginCmpPath = 1;
        endCmpPath = strlen(currentPath);
    }

    for(int i = beginCmpPath; i < endCmpPath ; i++ )
    {
        if ('/' == currentPath[i])
        {
            currentPath[i] = '\0';
            if (access(currentPath, 0) != 0)
            {
                if (mkdir(currentPath, 0755) == -1)
                {
                    printf("currentPath = %s\n", currentPath);
                    perror("mkdir error %s\n");
                    return -1;
                }
            }
            currentPath[i] = '/';
        }
    }
    return 0;
}

int main(int argc, char*argv[])
{
	printf(KGRN "TF ver: %s, TFLITE_SCHEMA_VERSION: %d\n" KNRM, TF_VERSION_STRING, TFLITE_SCHEMA_VERSION);
    uint32_t offset = 0, totalSize = 0, fileCnt = 0, totalLines, fileCntEnd, num_threads = 1, argIdx = 1;
    FILE *fpw = NULL;
    char desc[2048+1]   = {0};
    char buffer[2048+1] = {0};

    printf(KYEL "\nUsage: ./%s model list [offset size num_threads]", strrchr(argv[0], '/'));
    if (argc < 2)
    {
        printf(KRED "pls check your params\n" KNRM);
        exit(-1);
    }

    const char * model    = (const char *)argv[argIdx++];
    const char *pFileList = (const char *)argv[argIdx++];
    if (argc > argIdx) offset      = atoi(argv[argIdx++]);
    if (argc > argIdx) totalSize   = atoi(argv[argIdx++]);
    if (argc > argIdx) num_threads = atoi(argv[argIdx++]);
    printf(KGRN "model:%s list:%s offset:%u size:%u num_threads:%u\n" KNRM, model, pFileList, offset, totalSize, num_threads);

    std::unique_ptr<tflite::FlatBufferModel> tfmodel = tflite::FlatBufferModel::BuildFromFile(model);
    if(!tfmodel)
    {
        printf("Failed to mmap model, %s\n", model);
        exit(-1);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*tfmodel.get(), resolver)(&interpreter);
    if (!interpreter)
    {
        printf("Failed to construct interpreter\n");
        exit(-1);
    }

    interpreter->UseNNAPI(false);
    interpreter->SetAllowFp16PrecisionForFp32(false);
    interpreter->SetNumThreads(num_threads);

    std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
    std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
    std::cout << "inputs: " << interpreter->inputs().size() << "\n";
    std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";
    std::cout << "output(0) name: " << interpreter->GetOutputName(0) << "\n";

    int t_size = interpreter->tensors_size();

    for (int i = 0; i < t_size; i++)
    {
      if (interpreter->tensor(i)->name)
        std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Failed to allocate tensors!" << "\n";
        exit(-1);
    }

    int inputIdx = interpreter->inputs()[0];
    printf("input: %d\n", inputIdx);
    TfLiteIntArray* dims = interpreter->tensor(inputIdx)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];
    printf("Model In: [%d %d %d]\n", wanted_channels, wanted_height, wanted_width);

	const std::vector<int> inputs = interpreter->inputs();
	const std::vector<int> outputs = interpreter->outputs();

    std::cout << "number of inputs: " << inputs.size() << "\n";
    std::cout << "number of outputs: " << outputs.size() << "\n";
    printf("input tensor type: %d in [%d(kTfLiteFloat32), %d(kTfLiteUInt8)]\n", interpreter->tensor(inputIdx)->type, kTfLiteFloat32, kTfLiteUInt8);
	switch (interpreter->tensor(inputIdx)->type)
	{
		case kTfLiteFloat32:
		  break;
		case kTfLiteUInt8:
		  break;
		default:
		  std::cout << "cannot handle input type " << interpreter->tensor(inputIdx)->type << " yet" << "\n";
		  exit(-1);
	}

    struct timeval startTime, curTime;

    float* input = interpreter->typed_input_tensor<float>(0);
    float* output = interpreter->typed_output_tensor<float>(0);
    uint32_t sz = 128;

    ifstream inlist;
    inlist.open(pFileList);
    if(!inlist.is_open())
    {
        printf("open list %s failed\n", pFileList);
        return -1;
    }
    totalLines = std::count(std::istreambuf_iterator<char>(inlist), std::istreambuf_iterator<char>(), '\n');
    inlist.seekg(0, ios::beg);
    printf(KPURP "Total  %d\n" KNRM, totalLines);

    if (totalSize > 0)
        fileCntEnd = offset + totalSize;
    else
        fileCntEnd = totalLines - offset;
    fileCntEnd = std::min(fileCntEnd, totalLines);

    printf("start process list, fileCntEnd: %d\n", fileCntEnd);
    gettimeofday(&startTime, nullptr);
    std::string line;
    while (getline(inlist, line))
    {
        if (fileCnt < offset)
        {
            fileCnt++;
            continue;
        }

        if (fileCnt == offset)
            printf(KYEL "start from %d, %s\n" KNRM, fileCnt, line.c_str());

        struct timeval beg, end;

        gettimeofday(&beg, NULL);

        char *imgFile = (char*)line.c_str();
        cv::Mat img = cv::imread(imgFile, 0);
        from_y_normal(img.data, img.cols, img.rows, input, 127.5, 0.0078125, num_threads);

        interpreter->Invoke();

        gettimeofday(&end, NULL);

        gettimeofday(&curTime, NULL);

        #define ITER_CNT 2000000
        if (0 == fileCnt%ITER_CNT)
        {
            if (NULL != fpw)
            {
                printf("\ndump %s", buffer);
                fflush(stdout);
                fclose(fpw);
                fpw = NULL;
                printf(KGRN " [ok]\n" KNRM);
            }
            sprintf(buffer, "/sdcard/lj/feature_3530/range-%d-%d.bin", fileCnt, std::min(fileCnt+ITER_CNT-1, totalLines-1));
            makeDir(buffer);
            if (NULL == (fpw = fopen(buffer,"wb")))
            {
                printf("open output error, %s!\n", buffer);
                exit(-1);
            }
        }

        if (NULL != fpw)
            fwrite(output, sz*sizeof(float), 1, fpw);
        else /* for offset not start from 0 */
        {
            sprintf(buffer, "/sdcard/lj/feature_3530/range-%d-%d.bin", fileCnt, std::min(fileCnt+ITER_CNT-1, totalLines-1));
            makeDir(buffer);
            if (NULL == (fpw = fopen(buffer,"wb")))
            {
                printf("open output error, %s!\n", buffer);
                exit(-1);
            }
            fwrite(output, sz*sizeof(float), 1, fpw);
        }

        if (fileCnt == offset)
            showResult(output, 128);

        sprintf(desc, "%d %d %u dump to %s %80s %d %d %d time: %.3f ms (%lu s) threads: %d", offset, totalLines, sz, buffer, imgFile,
                img.channels(), img.cols, img.rows,
                (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0),
                (curTime.tv_sec*1000000ul + curTime.tv_usec - startTime.tv_sec*1000000ul - startTime.tv_usec)/(1000000ul), num_threads);
        img.release();
        ++fileCnt;
        print_progressbar(50, fileCnt - offset, fileCntEnd - offset, desc);

        if (fileCnt >= fileCntEnd)
        {
            printf(KYEL "stop at %d, %s\n" KNRM, fileCnt, line.c_str());
            if (NULL != fpw)
            {
                printf("\ndump %s", buffer);
                fflush(stdout);
                fclose(fpw);
                fpw = NULL;
                printf(KGRN " [ok]\n" KNRM);
            }
            break;
        }
	}

    inlist.close();
    return 0;
}
