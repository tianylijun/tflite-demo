#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

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

int main(int argc, char*argv[])
{
	int numThreads = 1;
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
    interpreter->SetNumThreads(numThreads);

    std::cout << "tensors size: " << interpreter->tensors_size() << "\n";
    std::cout << "nodes size: " << interpreter->nodes_size() << "\n";
    std::cout << "inputs: " << interpreter->inputs().size() << "\n";
    std::cout << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
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

    struct timeval beg, end;
    gettimeofday(&beg, nullptr);
    int loopCnt = 10;

    float* input = interpreter->typed_input_tensor<float>(0);

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
           numThreads,
           output);

    return 0;
}
