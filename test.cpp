#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>


bool isASCII(const std::string& filePath, std::string& firstLine) {
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return false;
    }

    std::getline(file, firstLine);

    // Check if the first line contains the "solid" keyword
    bool isAsciiFormat = (firstLine.find("solid") != std::string::npos);

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
        verts[3 * i] = {v1[0], v1[1], v1[2]};
        verts[3 * i + 1] = {v2[0], v2[1], v2[2]};
        verts[3 * i + 2] = {v3[0], v3[1], v3[2]};

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
        std::vector<double> extractedColumns = {row[5], row[3], (row[4]+ZOFFSET)};
        std::vector<double> extractedColumns2 = {row[2], row[0], row[1]};
        nvecpoints.push_back(extractedColumns);
        Nvecs.push_back(extractedColumns2);
        points.push_back(extractedColumns);
    }
    for (const auto& row1 : verts) {
        std::vector<double> extractedColumns3 = {row1[2], row1[0], (row1[1]+ZOFFSET)};
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

inline double
max (double a, double b, double c)
{
  return (a > b) ? (a > c ? a : c) : (b > c ? b : c);
}

inline double
min (double a, double b, double c)
{
  return (a < b) ? (a < c ? a : c) : (b < c ? b : c);
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

    return {xq, yq};
}

void extractXYVectors(const std::vector<std::vector<double>>& meshgrid, std::vector<double>& xq, bool isy) {
    if (meshgrid.empty() || meshgrid[0].empty()) {
        std::cerr << "Invalid meshgrid." << std::endl;
        return;
    }

    int numRows = meshgrid.size();
    int numCols = meshgrid[0].size();

    // Extract xq and yq vectors
    xq.clear();
    if (isy){
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j < numCols; ++j) {
                if (j == 1) {
                    // Assuming the second column represents y values
                    xq.push_back(meshgrid[i][j]);
                }
            }
        }
    }
    else if (isy == false){
        xq=meshgrid[0];
    }
}


void interpolateData(const std::vector<std::vector<double>>& inputPoints, const std::vector<double>& inputValues,
                     const std::vector<std::vector<double>>& xq, const std::vector<std::vector<double>>& yq,
                     std::vector<std::vector<double>>& outputValues) {
    // Convert vectors of vectors to cv::Mat
    cv::Mat inputPointsMat(inputPoints.size(), inputPoints[0].size(), CV_64F);
    for (size_t i = 0; i < inputPoints.size(); ++i) {
        for (size_t j = 0; j < inputPoints[i].size(); ++j) {
            inputPointsMat.at<double>(i, j) = inputPoints[i][j];
        }
    }

    cv::Mat inputValuesMat(inputValues);
    cv::Mat xqMat(xq.size(), xq[0].size(), CV_64F);
    cv::Mat yqMat(yq.size(), yq[0].size(), CV_64F);

    for (size_t i = 0; i < xq.size(); ++i) {
        for (size_t j = 0; j < xq[i].size(); ++j) {
            xqMat.at<double>(i, j) = xq[i][j];
            yqMat.at<double>(i, j) = yq[i][j];
        }
    }

    // Perform interpolation using OpenCV
    cv::Mat outputValuesMat;
    cv::remap(inputValuesMat, outputValuesMat, xqMat, yqMat, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Convert cv::Mat to vector of vectors
    outputValues.clear();
    for (int i = 0; i < outputValuesMat.rows; ++i) {
        std::vector<double> row;
        for (int j = 0; j < outputValuesMat.cols; ++j) {
            row.push_back(outputValuesMat.at<double>(i, j));
        }
        outputValues.push_back(row);
    }
}

// Write vector of vectors to a file
void writeToFile(const std::string& filename, const std::vector<std::vector<double>>& data) {
    std::ofstream outFile(filename);

    if (outFile.is_open()) {
        for (const auto& row : data) {
            for (const auto& value : row) {
                outFile << value << " ";
            }
            outFile << "\n";
        }

        outFile.close();
        std::cout << "Data written to file: " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}

// Bilinear interpolation function
double interpolate(double q11, double q21, double q12, double q22, double x1, double x2, double y1, double y2, double x, double y) {
    double r1 = (x2 - x) / (x2 - x1) * q11 + (x - x1) / (x2 - x1) * q21;
    double r2 = (x2 - x) / (x2 - x1) * q12 + (x - x1) / (x2 - x1) * q22;
    return (y2 - y) / (y2 - y1) * r1 + (y - y1) / (y2 - y1) * r2;
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

    bool stretch_down = false ; //True: stretch paths toward the edge of the XY-coordinate-region of the part, False: stretch paths toward the bottom

    bool clip_paths_every_iteration = true; //true: trim paths every iteration (slower, but less errors), false: trim paths after all iterations (faster)

    double support_temp = 110; //support material extruder temperature (°C)
    double mesh_temp = 93.334; //upper layer material extruder temperature (°C)
    support_temp=this_is_america(support_temp);
    mesh_temp=this_is_america(mesh_temp);

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
    num_contourlayers = contourthickness/contourlayerheight;
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
    std::vector<std::vector<double>> lims = {{(stretchBX+4.0), (stretchFX-4.0)},
                                            {(stretchBY+4.0), (stretchFY-4.0)}};
    std::vector<std::vector<std::vector<double>>> contourlayers;
    std::vector<std::vector<double>> layers(INTnum_contourlayers, std::vector<double>(1));
    std::vector<std::vector<double>> Lpoints(nvecpoints.size(), std::vector<double>(nvecpoints[0].size(), 0));
    std::vector<std::vector<double>> Lnegative(nvecpoints.size(), std::vector<double>(nvecpoints[0].size(), 0));
    for (int k = 0; k <= INTnum_contourlayers; ++k) {
        for (size_t i = 0; i < nvecpoints.size(); ++i) {
            for (size_t j = 0; j < nvecpoints[0].size(); ++j) {
                Lnegative[i][j] = Nvecs[i][j] * k * contourlayerheight ;
                Lpoints[i][j] = nvecpoints[i][j] - Lnegative[i][j];
            }
        }
        contourlayers.push_back(concatenateMatrices(Lpoints, Nvecs));
    }
    float pointspacing;
    pointspacing = stepsize_contour_border;
    // Call the function to get the first 2D matrix
    std::vector<std::vector<double>> printshape = contourlayers[0];
    auto meshgrid = createMeshgrid(printshape, pointspacing);
    std::vector<std::vector<double>> xq;
    std::vector<std::vector<double>> yq;
    xq=meshgrid.first;
    yq=meshgrid.second;
    std::vector<double> vectxq, vectyq;
    extractXYVectors(xq, vectxq, false);
    extractXYVectors(yq, vectyq, true);
    std::vector<std::vector<double>> zq ;
    //(xq.size(), std::vector<double>(xq[0].size(), 0));
    std::vector<std::vector<double>> printshape2;
    std::vector<double> zprintshape;
    // Iterate through the original vector and extract the first three columns
    for (const auto& row : printshape) {
        if (row.size() >= 3) {
            printshape2.push_back({row[0], row[1]});
            zprintshape.push_back(row[2]);
        } else {
            // Handle the case where the row has fewer than three columns
            std::cerr << "Warning: Row has fewer than three columns." << std::endl;
        }
    }
    interpolateData(printshape2, zprintshape, xq, yq, zq);
    //griddata(printshape3[0], printshape3[1], printshape3[2], xq, yq, zq);
    writeToFile("interpolated_values.txt", zq);
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
    } else {
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

