#pragma once

#include "pch.h"
#include <collection.h>
#include <ppl.h>
#include <amp.h>
#include <amp_math.h>
#include "CNTKLibrary.h"

using namespace Platform;

namespace CNTKLib
{
    public ref class CNTKModel sealed
    {
		CNTK::FunctionPtr Model;

    public:
        CNTKModel(String^ modelFile);
		float GetPrediction(float input1, float input2);
    };
}
