#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include <opencv2/opencv.hpp>
#include <arm_neon.h>

using namespace std;
using namespace cv;
using namespace tflite;

#define KDRED "\u001b[31m"
#define KBLUH "\u001b[38;5;6m"
#define KNRM  "\033[0m"
#define KPURP "\033[35m"
#define KRED  "\033[31m"
#define KGRN  "\033[32m"
#define KGRND "\033[36m"
#define KYEL  "\033[33m"
#define KBLU  "\033[34m"

static void print_progressbar(unsigned int barLen, unsigned int curPos, unsigned int total, const char *desc)
{
    printf("\r");
    curPos++;
    int offset  = curPos*barLen/total;
    for(int i = 0; i < offset; i++)
        printf(KGRN "%s" KNRM, "■");
    for (int i = offset; i < barLen; i++)
        printf("■");
    printf(" [%.2f%%] %d/%d" KGRN "[%s]" KNRM, curPos*100.0/total, curPos, total, desc);
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

int main(int argc, char*argv[])
{
    printf(KGRN "TF ver: %s, TFLITE_SCHEMA_VERSION: %d\n" KNRM, TF_VERSION_STRING, TFLITE_SCHEMA_VERSION);
    int num_threads = 1;
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(argv[1]);
    if(!model)
    {
        printf("Failed to mmap model, %s\n", argv[1]);
        exit(0);
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
    if (!interpreter)
    {
        printf("Failed to construct interpreter\n");
        exit(0);
    }

    interpreter->UseNNAPI(false);
    interpreter->SetAllowFp16PrecisionForFp32(false);
    interpreter->SetNumThreads(num_threads);

    std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
    std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
    int t_size = interpreter->tensors_size();

    const std::vector<int> inputs = interpreter->inputs();
    const std::vector<int> outputs = interpreter->outputs();

    std::cout << "number of inputs: " << inputs.size() << "\n";
    std::cout << "number of outputs: " << outputs.size() << "\n";
    std::vector<int> insSize;
    std::vector<int> outsSize;
    printf("-------------------\n");
    int inIdx = 0;
    for (auto in_tensor_idx: inputs)
    {
        int in_size = 1;
        printf("In:%d name:%s\n", in_tensor_idx, interpreter->GetInputName(inIdx++));
        int indims = NumDimensions(interpreter->tensor(in_tensor_idx));
        for (int i = 0; i < indims; ++i)
        {
            in_size *= SizeOfDimension(interpreter->tensor(in_tensor_idx), i);
            if (0 == i)
                printf("[");
            printf(" %d", SizeOfDimension(interpreter->tensor(in_tensor_idx), i));
        }
        printf("] %d\n", in_size);
        insSize.push_back(in_size);
    }
    printf("-------------------\n");
    int outIdx = 0;
    for (auto out_tensor_idx: outputs)
    {
        int out_size = 1;
        printf("Out:%d name:%s\n", out_tensor_idx, interpreter->GetOutputName(outIdx++));
        int outdims = NumDimensions(interpreter->tensor(out_tensor_idx));
        for (int i = 0; i < outdims; ++i)
        {
            out_size *= SizeOfDimension(interpreter->tensor(out_tensor_idx), i);
            if (0 == i)
                printf("[");
            printf(" %d", SizeOfDimension(interpreter->tensor(out_tensor_idx), i));
        }
        printf("] %d\n", out_size);
        outsSize.push_back(out_size);
    }
    printf("-------------------\n");
#if 0
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        std::cout << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
#endif
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Failed to allocate tensors!" << "\n";
        exit(0);
    }

    int inputIdx = interpreter->inputs()[0];
    printf("input: %d\n", inputIdx);
    TfLiteIntArray* dims = interpreter->tensor(inputIdx)->dims;
    int wanted_height = dims->data[1];
    int wanted_width = dims->data[2];
    int wanted_channels = dims->data[3];
    printf("Model In: [%d %d %d]\n", wanted_channels, wanted_height, wanted_width);

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

    struct timeval beg, end;
    gettimeofday(&beg, nullptr);
    int loopCnt = 1;

    float* input = interpreter->typed_input_tensor<float>(0);
    cv::Mat img = cv::imread("/sdcard/lj/112.png", 0);
    from_y_normal(img.data, img.cols, img.rows, input, 127.5, 0.0078125, num_threads);
    img.release();

    for (int loop = 0 ; loop < loopCnt; loop++)
    {
        interpreter->Invoke();
        print_progressbar(50, loop, loopCnt, " ");
	}

    float* output = interpreter->typed_output_tensor<float>(0);

    gettimeofday(&end, nullptr);
    printf("\ntime: %ld ms, avg time : %.3f ms, loop: %d threads: %d, output:%p\n", 
           (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/1000, 
           (end.tv_sec*1000000 + end.tv_usec - beg.tv_sec*1000000 - beg.tv_usec)/(1000.0*loopCnt),
           loopCnt,
           num_threads,
           output);

    showResult(output, outsSize[0]);
    return 0;
}
