// Adapted from this example: https://github.com/Microsoft/CNTK/tree/release/latest/Examples/Evaluation/UWPImageRecognition

#include "pch.h"
#include "CNTKModel.h"
#include "CNTKLibrary.h"

#include <ppltasks.h>
#include <concurrent_vector.h>

using namespace CNTKLib;
using namespace Platform;
using namespace Microsoft::MSR::CNTK;
using namespace CNTK;

CNTKLib::CNTKModel::CNTKModel(String^ modelFile)
{
	std::wstring modelStringW(modelFile->Data());
	Model = CNTK::Function::Load(modelStringW);
}

float CNTKLib::CNTKModel::GetPrediction(float input1, float input2)
{
	const size_t inputDim = 2;

	auto evalDevice = CNTK::DeviceDescriptor::UseDefaultDevice();

	auto input = InputVariable({ inputDim }, DataType::Float, L"features");
	auto inputVar = Model->Arguments()[0];
	std::vector<float> data = { input1, input2 };

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> inputLayer = {};

	auto features = CNTK::Value::CreateBatch<float>(inputVar.Shape(), data, evalDevice, false);
	inputVar = Model->Arguments()[0];
	inputLayer.insert({ inputVar, features });

	std::unordered_map<CNTK::Variable, CNTK::ValuePtr> outputLayer = {};
	auto outputVar = Model->Output();
	auto outputShape = outputVar.Shape();
	auto possibleClasses = outputShape.Dimensions()[0];

	std::vector<float> rawOutputs(possibleClasses);
	auto outputs = CNTK::Value::CreateBatch<float>(outputShape, rawOutputs, evalDevice, false);
	outputLayer.insert({ outputVar, NULL });

	Model->Evaluate(inputLayer, outputLayer, evalDevice);

	auto outputVal = outputLayer[outputVar];
	std::vector<std::vector<float>> resultsWrapper;

	outputVal.get()->CopyVariableValueTo(outputVar, resultsWrapper);

	return resultsWrapper[0][0];
}