#include <vtkSmartPointer.h>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPoints.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <iostream>
#include <cmath>

bool compareDataArrays(vtkDataArray* array1, vtkDataArray* array2, double tolerance) {
    if (array1->GetNumberOfTuples() != array2->GetNumberOfTuples() ||
        array1->GetNumberOfComponents() != array2->GetNumberOfComponents()) {
        return false;
    }

    vtkIdType numTuples = array1->GetNumberOfTuples();
    int numComponents = array1->GetNumberOfComponents();

    for (vtkIdType i = 0; i < numTuples; ++i) {
        for (int j = 0; j < numComponents; ++j) {
            double value1 = array1->GetComponent(i, j);
            double value2 = array2->GetComponent(i, j);
            if (std::fabs(value1 - value2) > tolerance) {
                std::cout << "Difference at tuple " << i << ", component " << j 
                          << ": " << value1 << " vs " << value2 << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool compareVTKFiles(const char* file1, const char* file2, double tolerance) {
    // Create readers for both files
    vtkSmartPointer<vtkStructuredPointsReader> reader1 = vtkSmartPointer<vtkStructuredPointsReader>::New();
    reader1->SetFileName(file1);
    reader1->Update();

    vtkSmartPointer<vtkStructuredPointsReader> reader2 = vtkSmartPointer<vtkStructuredPointsReader>::New();
    reader2->SetFileName(file2);
    reader2->Update();

    // Get the data from both files
    vtkSmartPointer<vtkStructuredPoints> data1 = reader1->GetOutput();
    vtkSmartPointer<vtkStructuredPoints> data2 = reader2->GetOutput();

    // Get the point data from both files
    vtkSmartPointer<vtkPointData> pointData1 = data1->GetPointData();
    vtkSmartPointer<vtkPointData> pointData2 = data2->GetPointData();

    if (pointData1->GetNumberOfArrays() != pointData2->GetNumberOfArrays()) {
        std::cout << "Number of data arrays differ." << std::endl;
        return false;
    }

    // Compare each data array
    for (int i = 0; i < pointData1->GetNumberOfArrays(); ++i) {
        vtkDataArray* array1 = pointData1->GetArray(i);
        vtkDataArray* array2 = pointData2->GetArray(i);

        if (!compareDataArrays(array1, array2, tolerance)) {
            std::cout << "Data arrays " << array1->GetName() << " and " << array2->GetName() << " differ." << std::endl;
            return false;
        }
    }

    return true;
}

int main(int argc, char *argv[]) {
    // Verify input arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " file1.vtk file2.vtk tolerance" << std::endl;
        return EXIT_FAILURE;
    }

    const char* file1 = argv[1];
    const char* file2 = argv[2];
    double tolerance = std::stod(argv[3]);

    // Compare the files
    bool areEqual = compareVTKFiles(file1, file2, tolerance);
    std::cout << "The files are " << (areEqual ? "similar within the tolerance" : "[not] similar within the tolerance") << std::endl;

    return EXIT_SUCCESS;
}
