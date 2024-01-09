#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>  // Include the highgui module useless btw


using namespace cv;
using namespace std;

const double NaN = std::numeric_limits<double>::quiet_NaN();

bool isASCII(const std::string& filePath, std::string& firstLine) {
    std::ifstream file(filePath);

    std::getline(file, firstLine);

    // Check if the first line contains the "solid" keyword
    bool isAsciiFormat = (firstLine.find("solid") != std::string::npos);

    // Check if the file exists
    if (!file) {
        std::cerr << "Error: File does not exist: " << filePath << std::endl;
        return true;
    }

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return true;
    }
    file.close();
    return isAsciiFormat;
}

void readSTL(const std::string& file, float angleomit,
    std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& Nvecs,
    std::vector<std::vector<double>>& nvecpoints,
    float ZOFFSET) {

    // Open the file
    std::ifstream fileStream(file, std::ios::binary | std::ios::ate);

    // Get the file size
    std::streamsize fileSize = fileStream.tellg();
    fileStream.seekg(0, std::ios::beg);

    // Read the entire file into a vector
    std::vector<uint8_t> M(fileSize);
    fileStream.read(reinterpret_cast<char*>(M.data()), fileSize);
    fileStream.close();
    // Extract relevant information
    std::vector<uint8_t> info(M.begin() + 84, M.end());
    uint32_t nFaces = *reinterpret_cast<uint32_t*>(&M[80]);

    // Initialize vectors
    std::vector<std::vector<float>> nvecs(nFaces, std::vector<float>(6, 0.0f));
    std::vector<std::vector<float>> verts(3 * nFaces, std::vector<float>(3, 0.0f));

    for (uint32_t i = 0; i < nFaces; ++i) {
        std::vector<uint8_t> facet(info.begin() + 50 * i, info.begin() + 50 * (i + 1));

        std::vector<float> v1(3, 0.0f);
        std::vector<float> v2(3, 0.0f);
        std::vector<float> v3(3, 0.0f);
        if (facet.size() >= 48) {
            // v1
            std::memcpy(&v1[0], &facet[12], sizeof(float));
            std::memcpy(&v1[1], &facet[16], sizeof(float));
            std::memcpy(&v1[2], &facet[20], sizeof(float));

            // v2
            std::memcpy(&v2[0], &facet[24], sizeof(float));
            std::memcpy(&v2[1], &facet[28], sizeof(float));
            std::memcpy(&v2[2], &facet[32], sizeof(float));

            // v3
            std::memcpy(&v3[0], &facet[36], sizeof(float));
            std::memcpy(&v3[1], &facet[40], sizeof(float));
            std::memcpy(&v3[2], &facet[44], sizeof(float));
        }
        verts[3 * i] = { v1[0], v1[1], v1[2] };
        verts[3 * i + 1] = { v2[0], v2[1], v2[2] };
        verts[3 * i + 2] = { v3[0], v3[1], v3[2] };

        nvecs[i][0] = *reinterpret_cast<float*>(&facet[0]);
        nvecs[i][1] = *reinterpret_cast<float*>(&facet[4]);
        nvecs[i][2] = *reinterpret_cast<float*>(&facet[8]);
        nvecs[i][3] = (v1[0] + v2[0] + v3[0]) / 3;
        nvecs[i][4] = (v1[1] + v2[1] + v3[1]) / 3;
        nvecs[i][5] = (v1[2] + v2[2] + v3[2]) / 3;
    }
    // Filter nvecs
    std::vector<std::vector<float>> filteredNvecs;
    for (const auto& nvec : nvecs) {
        if (nvec[1] > 0 &&
            atan(nvec[1] / sqrt(nvec[0] * nvec[0] + nvec[2] * nvec[2])) > angleomit * M_PI / 180) {
            filteredNvecs.push_back(nvec);
        }
    }

    for (const auto& row : filteredNvecs) {
        std::vector<double> extractedColumns = { row[5], row[3], (row[4] + ZOFFSET) };
        std::vector<double> extractedColumns2 = { row[2], row[0], row[1] };
        nvecpoints.push_back(extractedColumns);
        Nvecs.push_back(extractedColumns2);
        points.push_back(extractedColumns);
    }
    for (const auto& row1 : verts) {
        std::vector<double> extractedColumns3 = { row1[2], row1[0], (row1[1] + ZOFFSET) };
        points.push_back(extractedColumns3);
    }
}

void negateFirstColumn(std::vector<std::vector<double>>& matrix) {
    for (auto& row : matrix) {
        row[0] = -row[0];
    }
}

double this_is_america(double celsius) {
    return (celsius * 9 / 5) + 32;
}

double goodcolumn(const std::vector<std::vector<double>>& matrix, size_t column) {
    if (matrix.empty() || matrix[0].size() <= column) {
        std::cerr << "Error: Invalid matrix or column index.\n";
        return 0.0; // Return a default value or handle the error as needed
    }
    else {
        return 1.0;
    }
}

double calculateMean(const std::vector<std::vector<double>>& matrix, size_t column) {
    goodcolumn(matrix, column);
    double sum = 0.0;
    for (const auto& row : matrix) {
        sum += row[column];
    }

    return sum / matrix.size();
}
double calculateMax(const std::vector<std::vector<double>>& matrix, size_t column) {
    goodcolumn(matrix, column);
    double maxVal = matrix[0][column];
    for (const auto& row : matrix) {
        maxVal = std::max(maxVal, row[column]);
    }

    return maxVal;
}
double calculateMin(const std::vector<std::vector<double>>& matrix, size_t column) {
    goodcolumn(matrix, column);
    double minVal = matrix[0][column];
    for (const auto& row : matrix) {
        minVal = std::min(minVal, row[column]);
    }

    return minVal;
}

std::vector<std::vector<double>> concatenateMatrices(const std::vector<std::vector<double>>& matrix1,
    const std::vector<std::vector<double>>& matrix2) {
    // Check if the matrices have the same number of rows
    if (matrix1.size() != matrix2.size()) {
        std::cerr << "Matrices have different numbers of rows and cannot be concatenated." << std::endl;
        return std::vector<std::vector<double>>();  // Return an empty matrix
    }
    // Concatenate matrices horizontally
    std::vector<std::vector<double>> concatenatedMatrix;
    for (size_t i = 0; i < matrix1.size(); ++i) {
        // Combine the rows of both matrices
        std::vector<double> combinedRow;
        combinedRow.insert(combinedRow.end(), matrix1[i].begin(), matrix1[i].end());
        combinedRow.insert(combinedRow.end(), matrix2[i].begin(), matrix2[i].end());

        // Add the combined row to the concatenated matrix
        concatenatedMatrix.push_back(combinedRow);
    }

    return concatenatedMatrix;
}

std::vector<std::vector<double>> convertToDoubleVector(const std::vector<std::vector<float>>& floatVector) {
    std::vector<std::vector<double>> doubleVector;
    for (const auto& row : floatVector) {
        std::vector<double> convertedRow;
        for (const auto& value : row) {
            if (std::isnan(value)) {
                convertedRow.push_back(std::numeric_limits<double>::quiet_NaN());
            } else {
                convertedRow.push_back(static_cast<double>(value));
            }
        }
        doubleVector.push_back(convertedRow);
    }
    return doubleVector;
}

// Define a function to create a meshgrid
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> createMeshgrid(
    const std::vector<std::vector<double>>& data,
    double pointSpacing) {
    // Determine the range
    double minX = data[0][0];
    double maxX = data[0][0];
    double minY = data[0][1];
    double maxY = data[0][1];

    for (const auto& point : data) {
        minX = std::min(minX, point[0]);
        maxX = std::max(maxX, point[0]);
        minY = std::min(minY, point[1]);
        maxY = std::max(maxY, point[1]);
    }

    // Create meshgrid
    std::vector<std::vector<double>> xq;
    std::vector<std::vector<double>> yq;

    for (double y = minY; y <= maxY; y += pointSpacing) {
        std::vector<double> rowXq;
        std::vector<double> rowYq;

        for (double x = minX; x <= maxX; x += pointSpacing) {
            rowXq.push_back(x);
            rowYq.push_back(y);
        }

        xq.push_back(rowXq);
        yq.push_back(rowYq);
    }

    return { xq, yq };
}


void extractXYVectors(const std::vector<std::vector<double>>& meshgrid,
    std::vector<double>& xq, bool isy) {
    if (meshgrid.empty() || meshgrid[0].empty()) {
        std::cerr << "Invalid meshgrid." << std::endl;
        return;
    }

    int numRows = meshgrid.size();
    int numCols = meshgrid[0].size();

    // Extract xq and yq vectors
    xq.clear();
    if (isy) {
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                if (j == 1) {
                    // Assuming the second column represents y values
                    xq.push_back(meshgrid[i][j]);
                }
            }
        }
    }
    else if (isy == false) {
        xq = meshgrid[0];
    }
}

std::vector<std::vector<cv::Point2f>> convertToPoints(
    const std::vector<std::vector<double>>& xq,
    const std::vector<std::vector<double>>& yq) {
    // Ensure xq and yq have the same dimensions
    if (xq.size() != yq.size() || xq.empty() || xq[0].size() != yq[0].size()) {
        std::cerr << "Error: Input vectors have different dimensions." << std::endl;
        return std::vector<std::vector<cv::Point2f>>(); // Return an empty vector in case of error
    }

    // Create the points vector
    std::vector<std::vector<cv::Point2f>> points;

    // Iterate through the elements of xq and yq, creating Point2f objects
    for (size_t i = 0; i < xq.size(); ++i) {
        std::vector<cv::Point2f> row_points;
        for (size_t j = 0; j < xq[i].size(); ++j) {
            cv::Point2f point(static_cast<float>(xq[i][j]), static_cast<float>(yq[i][j]));
            row_points.push_back(point);
        }
        points.push_back(row_points);
    }

    return points;
}

std::vector<std::vector<cv::Point2f>> processMatrices(const std::vector<std::vector<double>>& dataMatrix,
                                                       const std::vector<std::vector<cv::Point2f>>& coordinatesMatrix) {
    std::vector<std::vector<cv::Point2f>> resultMatrix;

    for (size_t i = 0; i < dataMatrix.size(); ++i) {
        std::vector<cv::Point2f> resultRow;
        for (size_t j = 0; j < dataMatrix[i].size(); ++j) {
            if (dataMatrix[i][j] == 1) {
                resultRow.push_back(coordinatesMatrix[i][j]);
            } else {
                resultRow.push_back(cv::Point2f(NAN, NAN));
            }
        }
        resultMatrix.push_back(resultRow);
    }

    return resultMatrix;
}

void replaceOnesWithIncrementedValues(const std::vector<std::vector<double>>& inputMatrix,
std::vector<std::vector<float>>& outputMatrix,
const std::vector<float>& resultvec) {
    int replacementValue = 0;
    for (size_t i = 0; i < inputMatrix.size(); ++i) {
        for (size_t j = 0; j < inputMatrix[i].size(); ++j) {
            if (std::isnan(inputMatrix[i][j])) {
                // Handle NaN values in inputMatrix
                outputMatrix[i][j] = std::numeric_limits<float>::quiet_NaN();
            } else if (inputMatrix[i][j] == 1) {
				if (replacementValue < resultvec.size()) {
                	outputMatrix[i][j] = resultvec[replacementValue];
                	++replacementValue;
				}else {
					outputMatrix[i][j] = 1;
					std::cerr << "Error: replacementValue exceeds the total number of elements." << std::endl;
				}
            }
        }
    }
}


struct Point3D {
    float x;
    float y;
    float z;
};

// Custom hash function for std::pair<double, double>
struct PairHash {
    template <typename T1, typename T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);

        // Simple hash combining technique
        return h1 ^ h2;
    }
};

std::vector<std::vector<Point3D>> associateZValues(const std::vector<std::vector<double>>& printshape3,
                                                   const std::vector<std::vector<cv::Point2f>>& coordinates2D) {

    // Create and initialize coordinates3D
    std::vector<std::vector<Point3D>> coordinates3D(coordinates2D.size(),
                                                   std::vector<Point3D>(coordinates2D[0].size(),
                                                                        {0.0f, 0.0f, -std::numeric_limits<float>::infinity()}));

    // Create a map for quick lookups with custom hash function
    std::unordered_multimap<std::pair<double, double>, float, PairHash> printshape3Map;
    for (const auto& entry : printshape3) {
        printshape3Map.emplace(std::make_pair(entry[0], entry[1]), static_cast<float>(entry[2]));
    }

    // Associate z values
    for (size_t i = 0; i < coordinates2D.size(); ++i) {
        for (size_t j = 0; j < coordinates2D[i].size(); ++j) {
            double x2D = coordinates2D[i][j].x;
            double y2D = coordinates2D[i][j].y;
            // Find corresponding z value in printshape3Map
            auto range = printshape3Map.equal_range({x2D, y2D});
            float maxZValue = -std::numeric_limits<float>::infinity();

            for (auto it = range.first; it != range.second; ++it) {
                maxZValue = std::max(maxZValue, it->second);
            }

            // Convert and assign to Point3D
            coordinates3D[i][j] = {static_cast<float>(x2D), static_cast<float>(y2D), maxZValue};
            if (std::abs(x2D) < 1e-6 && std::abs(y2D) < 1e-6) {
                coordinates3D[i][j] = {static_cast<float>(x2D), static_cast<float>(y2D), 0.0f};
            }
        }
    }

    return coordinates3D;
}

// Custom function to replace -inf values in a row
void replaceInfValuesInRow(std::vector<cv::Point3f>& row) {
    float Infval = -std::numeric_limits<float>::infinity();
    std::vector<float> nonInfValues;

    // Collect non-infinite values in the row
    for (const auto& point : row) {
        if (point.z != Infval) {
            nonInfValues.push_back(point.z);
        }
    }

    // Replace -inf values based on the collected non-infinite values
    if (!nonInfValues.empty()) {
        float averageValue;
        if (nonInfValues.size() == 1) {
            averageValue = nonInfValues[0];
        } else if (nonInfValues.size() == 2) {
            averageValue = std::accumulate(nonInfValues.begin(), nonInfValues.end(), 0.0f) / 2;
        }

        for (auto& point : row) {
            if (point.z == Infval) {
                point.z = averageValue;
            }
        }
    }
}

// Function to replace -inf values in a matrix of Point3f
std::vector<std::vector<cv::Point3f>> replaceInfZValuesInMatrix(const std::vector<std::vector<cv::Point3f>>& coordinates3D) {
    std::vector<std::vector<cv::Point3f>> resultMatrix = coordinates3D;

    for (auto& row : resultMatrix) {
        replaceInfValuesInRow(row);
    }

    return resultMatrix;
}

std::vector<std::vector<cv::Point3f>> convertToCvPoint3f(const std::vector<std::vector<Point3D>>& input) {
    std::vector<std::vector<cv::Point3f>> output;

    for (const auto& row : input) {
        std::vector<cv::Point3f> convertedRow;
        for (const auto& point : row) {
            convertedRow.push_back(cv::Point3f(point.x, point.y, point.z));
        }
        output.push_back(convertedRow);
    }

    return output;
}

float triangle_area(float x1, float y1, float x2, float y2, float x3, float y3) {
    // Calculate the area of the triangle using the Shoelace Formula
    float area = 0.5 * std::fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
    return area;
}

void interpolation(std::vector<std::vector<cv::Point3f>>& matrix1 , std::vector<cv::Point2f>& points, std::vector<float>& interpolated){
    for (int i=0; i<points.size(); ++i) {
        float P1[3];
        float P2[3];
        float P3[3];
        float P[2];
        P[0] = points[i].x;
        P[1] = points[i].y;
        P1[0] = matrix1[i][0].x;
        P1[1] = matrix1[i][0].y;
        P1[2] = matrix1[i][0].z;
        P2[0] = matrix1[i][1].x;
        P2[1] = matrix1[i][1].y;
        P2[2] = matrix1[i][1].z;
        P3[0] = matrix1[i][2].x;
        P3[1] = matrix1[i][2].y;
        P3[2] = matrix1[i][2].z;
        float abc, pbc, apc, abp,v;
        abc = triangle_area ( P1[0], P1[1], P2[0], P2[1], P3[0], P3[1] );
        pbc = triangle_area ( P[0], P[1], P2[0],    P2[1],    P3[0],    P3[1] );
        apc = triangle_area ( P1[0],    P1[1],    P[0], P[1], P3[0],    P3[1] );
        abp = triangle_area ( P1[0],    P1[1],    P2[0],    P2[1],    P[0], P[1] );
        v= (pbc * P1[3] + apc * P2[3] + abp * P3[3])/ abc;
        interpolated.push_back(v);
    }
}

std::vector<std::vector<float>> pfillMatrices(const std::vector<std::vector<double>>& dataMatrix,
                                                       std::vector<float>& values) {
    std::vector<std::vector<float>> resultMatrix;
    int ij=0;
    for (size_t i = 0; i < dataMatrix.size(); ++i) {
        std::vector<float> resultRow;
        for (size_t j = 0; j < dataMatrix[i].size(); ++j) {
            if (dataMatrix[i][j] == 1) {
                if (ij < values.size()){
                    resultRow.push_back(values[ij]);
                    ++ij;
                }else {
                    resultRow.push_back(NAN);
                }
            } else {
                resultRow.push_back(NAN);
            }
        }
        resultMatrix.push_back(resultRow);
    }

    return resultMatrix;
}

void griddata(std::vector<std::vector<double>>& inputmat,
    std::vector<std::vector<double>>& xq,
    std::vector<std::vector<double>>& yq,
    std::vector<std::vector<double>>& zq, double pointSpacing)
{
    std::vector<std::vector<double>> printshape3;
    std::vector <double> thirdrawprintshape,firstrawprintshape, secondrawprintshape;
    // Iterate through the original vector and extract the first three columns
    for (const auto& row : inputmat) {
        if (row.size() >= 3) {
            printshape3.push_back({ row[0], row[1], row[2] });
            firstrawprintshape.push_back(row[0]);
            secondrawprintshape.push_back(row[1]);
            thirdrawprintshape.push_back(row[2]);

        }
        else {
            // Handle the case where the row has fewer than three columns
            std::cerr << "Warning: Row has fewer than three columns." << std::endl;
        }
    }
    auto meshgrid = createMeshgrid(inputmat, pointSpacing);

    xq = meshgrid.first;
    yq = meshgrid.second;

    std::vector<std::vector<cv::Point2f>> xqyqpoints = convertToPoints(xq, yq);
    std::vector<double> vectxq, vectyq;
    extractXYVectors(xq, vectxq, false);
    extractXYVectors(yq, vectyq, true);
    // Construct vector of cv::Point2f
    std::vector<cv::Point2f> gridpoints;
    for (size_t i = 0; i < firstrawprintshape.size(); ++i) {
        float x_valeue = static_cast<float>(firstrawprintshape[i]);
        float y_valeue = static_cast<float>(secondrawprintshape[i]);
        cv::Point2f pointi(x_valeue, y_valeue);
        gridpoints.push_back(pointi);
    }

    // Find the convex hull of the points
    std::vector<cv::Point2f> hull;
    cv::convexHull(gridpoints, hull);
    // Now, 'hull' contains the contour points of the geometrical figure

    std::vector<std::vector<double>> dataloca;

    // Iterate through the vectors of points and check if each point is inside the shape
    for (const auto& points : xqyqpoints) {
        std::vector<double> resultRow;
        for (const auto& point : points) {
            double distance = cv::pointPolygonTest(hull, point, true);

            if (distance > 0) {
                // The point is inside the shape
                resultRow.push_back(1.0);
            } else if (distance == 0) {
                // The point is on the contour
                resultRow.push_back(0.0);
            } else {
                // The point is outside the shape
                resultRow.push_back(NaN);
            }
        }
        dataloca.push_back(resultRow);
    }
    std::vector<std::vector<cv::Point2f>> coordloc = processMatrices(dataloca, xqyqpoints);
    //see in which delaunay triangle the point is
    cv::Rect boundingRect = cv::boundingRect(gridpoints);

    // Create a Subdiv2D object and insert the points
    cv::Subdiv2D subdiv(boundingRect);
    subdiv.insert(gridpoints);
    std::vector<cv::Vec4f> edgeList;
    subdiv.getEdgeList(edgeList);
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    std::vector<std::vector<cv::Point2f>> resultvertices;
    std::vector<cv::Point2f> pt;
    for (const auto &points : coordloc) {
        for (const auto &point : points) {
            // Check for NaN values and skip them
            if (!std::isnan(point.x) && !std::isnan(point.y)) {
                // Find the corresponding triangle using locate
                int edgeIndex, vertex;
                subdiv.locate(point, edgeIndex, vertex);
                cv::Vec6f spectriangle;
                int nextEdge = subdiv.getEdge(edgeIndex, cv::Subdiv2D::NEXT_AROUND_ORG);
                cv::Point2f vertices[3];
                std::vector<cv::Point2f> verticesVector;
                if (nextEdge > 0){
                    for (int i = 0; i < 3; ++i) {
                        int org = subdiv.edgeOrg(nextEdge, &vertices[i]);
                        nextEdge = subdiv.edgeDst(subdiv.nextEdge(nextEdge));
                        verticesVector.push_back(vertices[i]);
                    }
                    double distance = cv::pointPolygonTest(verticesVector, point, true);
                    if (distance > 0) {
                        resultvertices.push_back(verticesVector);
                        pt.push_back(point);
                    }
                }

            }
        }
    }
    std::vector<std::vector<Point3D>> coordinates3D = associateZValues(printshape3, resultvertices);
    std::vector<std::vector<cv::Point3f>> convertedCoordinates3D = convertToCvPoint3f(coordinates3D);
    std::vector<std::vector<cv::Point3f>> modified3DMatrix = replaceInfZValuesInMatrix(convertedCoordinates3D);
    std::vector<float> interpolatedvals;
    interpolation(modified3DMatrix, pt, interpolatedvals);
    std::vector<std::vector<float>> filledmat = pfillMatrices(dataloca, interpolatedvals);
    zq=convertToDoubleVector(filledmat);
}

void writeDoubleVectorToFile(const std::vector<std::vector<double>>& doubleVector, const std::string& filename) {
    std::ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    for (const auto& row : doubleVector) {
        for (const auto& value : row) {
            outputFile << value << " ";
        }
        outputFile << std::endl;
    }

    outputFile.close();
}

void SLICER_CALCULATIONS(
    std::vector<std::vector<double>>& points,
    std::vector<std::vector<double>>& Nvecs,
    std::vector<std::vector<double>>& nvecpoints)
{
    float stepsize_contour = 0.13; //sampling distance between points for contoured infill, mm
    float stepsize_contour_border = 0.05; //sampling distance between points for contoured borders, mm

    float layerheight = 0.3; //non-contoured layer height, mm (for planar layers)
    float linewidth = 0.4; //nozzlewidth, mm
    float close_linespacing = 0.4; //spacing between close contoured lines, mm (not upper contours)

    float support_interface_offset = 0.3; //Gap between support and upper contoured layers (mm)
    float sampling_dist = 0.01; //final point spacing for contoured lines, mm (interpolated at end of generation)

    float upper_layers_flowfactor = 3.3; //flow rate multiplier for upper contoured layers
    float upper_layers_borderfactor = 4; //flow rate multiplier for upper contoured layer borders

    float flowfactor = 1.3; //flow rate multiplier for all other layers

    bool stretch_down = false; //True: stretch paths toward the edge of the XY-coordinate-region of the part, False: stretch paths toward the bottom

    bool clip_paths_every_iteration = true; //true: trim paths every iteration (slower, but less errors), false: trim paths after all iterations (faster)

    double support_temp = 110; //support material extruder temperature (°C)
    double mesh_temp = 93.334; //upper layer material extruder temperature (°C)
    support_temp = this_is_america(support_temp);
    mesh_temp = this_is_america(mesh_temp);

    float topcontour_linespacing = 1.2; //upper layers spacing between paths (mm)
    float num = 25; //number of samples for running average smoothing (more = more smooth, less accurate)
    float filamentD = 1.75; //filament diameter (mm)

    int infillspacing = 3; //linespacing for infill/support material mm
    int skinlayer = 0; //number of layers of outer skin for the support material/planar layers
    int wall_lines = 1; //number of wall lines for the support material/planar layers
    int wallsmoothnum = 27; //number of samples for running average smoothing of walls for the support/planar layers  (more = more smooth, less accurate)

    bool flatbottom = true; //true: part sits flat on build plate, false: part is upper layers only (use for mesh lens)
    int bordersmoothnum = 40; //number of samples for running average smoothing for contoured borders (more = more smooth, less accurate)
    int contourborderlines = 3; //number of contoured border lines

    float contourlayerheight = 0.2; //contoured layers layer height, mm
    float contourthickness = 4; //total contoured thickness, mm
    float num_contourlayers; //number of contoured layers
    num_contourlayers = contourthickness / contourlayerheight;
    int INTnum_contourlayers = static_cast<int>(std::ceil(num_contourlayers));
    int num_topcontour = 2; //number of upper contoured layers (not support)
    int num_topborder = 2; //number of border lines in the upper contoured layers

    double middleX = calculateMean(points, 0);
    double middleY = calculateMean(points, 1);
    double stretchFX = calculateMax(points, 0) + 4.0;
    double stretchFY = calculateMax(points, 1) + 4.0;
    double stretchBX = calculateMin(points, 0) - 4.0;
    double stretchBY = calculateMin(points, 1) - 4.0;
    double clearZ = calculateMax(points, 2) + 4.0;
    std::vector<std::vector<double>> lims = { {(stretchBX + 4.0), (stretchFX - 4.0)},
                                            {(stretchBY + 4.0), (stretchFY - 4.0)} };
    std::vector<std::vector<std::vector<double>>> contourlayers;
    std::vector<std::vector<double>> layers(INTnum_contourlayers, std::vector<double>(1));
    std::vector<std::vector<double>> Lpoints(nvecpoints.size(), std::vector<double>(nvecpoints[0].size(), 0));
    std::vector<std::vector<double>> Lnegative(nvecpoints.size(), std::vector<double>(nvecpoints[0].size(), 0));
    for (int k = 0; k <= INTnum_contourlayers; ++k) {
        for (size_t i = 0; i < nvecpoints.size(); ++i) {
            for (size_t j = 0; j < nvecpoints[0].size(); ++j) {
                Lnegative[i][j] = Nvecs[i][j] * k * contourlayerheight;
                Lpoints[i][j] = nvecpoints[i][j] - Lnegative[i][j];
            }
        }
        contourlayers.push_back(concatenateMatrices(Lpoints, Nvecs));
    }
    float pointspacing;
    pointspacing = stepsize_contour_border;
    // Call the function to get the first 2D matrix
    std::vector<std::vector<double>> printshape = contourlayers[0];
    std::vector<std::vector<double>> xq, yq, zq;
    griddata(printshape, xq, yq, zq, pointspacing);
    writeDoubleVectorToFile(zq, "zq.txt");
    std::cout << "zq.txt" << std::endl;
}

int main() {
    std::string filePath = "EYE6.stl";  // Replace with your STL file path
    std::string firstLine;

    bool flipped = false;
    float ZOFFSET = 0.0; //vertical offset (mm)
    float angleOmit = 30.0f;  // Replace with your desired angle omit value

    if (isASCII(filePath, firstLine)) {
        std::cout << "The file is in ASCII format." << std::endl;
        std::cout << "Reformat the file to binary format and try again." << std::endl;
    }
    else {
        std::cout << "The file is in binary format. Good job, great format" << std::endl;
        std::vector<std::vector<double>> nvecpoints, Nvecs, points;
        readSTL(filePath, angleOmit, points, Nvecs, nvecpoints, ZOFFSET);
        if (flipped) {
            std::cout << "The object will be flipped" << std::endl;
            negateFirstColumn(points);
            negateFirstColumn(nvecpoints);
            negateFirstColumn(Nvecs);
        }
        SLICER_CALCULATIONS(points, Nvecs, nvecpoints);
    }

    return 0;
}

