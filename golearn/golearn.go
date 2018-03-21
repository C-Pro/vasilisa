package main

// RUN with GODEBUG=cgocheck=0 go run golearn.go

import (
	"fmt"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/linear_models"
)

func main() {
	data := [10][3]int16{
		[3]int16{156, 49, 1},
		[3]int16{180, 80, -1},
		[3]int16{175, 72, -1},
		[3]int16{170, 55, 1},
		[3]int16{179, 20, -1},
		[3]int16{158, 50, 1},
		[3]int16{183, 92, -1},
		[3]int16{179, 74, -1},
		[3]int16{168, 53, 1},
		[3]int16{177, 73, -1},
	}
	attrs := make([]base.Attribute, 3)
	attrs[0] = base.NewFloatAttribute("Height")
	attrs[1] = base.NewFloatAttribute("Weight")
	attrs[2] = base.NewFloatAttribute("Gender")

	// Create a new, empty DenseInstances
	newInst := base.NewDenseInstances()

	// Add the attributes
	newSpecs := make([]base.AttributeSpec, len(attrs))
	for i, a := range attrs {
		newSpecs[i] = newInst.AddAttribute(a)
	}

	newInst.Extend(len(data))
	for i, d := range data {
		// Allocate space
		//newInst.Extend(1)

		// Write the data
		newInst.Set(newSpecs[0], i, base.PackFloatToBytes(float64(d[0])))
		newInst.Set(newSpecs[1], i, base.PackFloatToBytes(float64(d[1])))
		newInst.Set(newSpecs[2], i, base.PackFloatToBytes(float64(d[2])))
	}
	newInst.AddClassAttribute(attrs[len(attrs)-1])

	cls, err := linear_models.NewLogisticRegression("l2", 1, 0.01)
	if err != nil {
		fmt.Printf("Err creating classifier: %v\n", err)
		return
	}

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(newInst, 0.50)
	fmt.Println(trainData)
	fmt.Println(testData)
	err = cls.Fit(trainData)
	if err != nil {
		fmt.Printf("Err training: %v\n", err)
		return
	}

	//predictions := cls.Predict(testData)
	predictions, err := cls.Predict(testData)
	if err != nil {
		fmt.Printf("Err predicting: %v\n", err)
		return
	}

	// Prints precision/recall metrics
	confusionMat, _ := evaluation.GetConfusionMatrix(testData, predictions)
	fmt.Println(evaluation.GetSummary(confusionMat))

}
