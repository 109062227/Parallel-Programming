#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip> // For setting precision

using namespace std;
int mode;
int height;
int width;
#define POINT 4

// Function to convert binary input file to text format
void convertBinaryToText(const char *binaryFile, const char *textFile)
{
    ifstream infile(binaryFile, ios::binary);
    if (!infile.is_open())
    {
        cerr << "Error: Could not open " << binaryFile << " for reading." << endl;
        exit(EXIT_FAILURE);
    }

    // Read dimensions
    int h, w, k;
    infile.read(reinterpret_cast<char *>(&h), sizeof(int));
    infile.read(reinterpret_cast<char *>(&w), sizeof(int));
    infile.read(reinterpret_cast<char *>(&k), sizeof(int));

    // Validate dimensions
    if (h <= 0 || w <= 0 || k <= 0 || k > min(h, w))
    {
        cerr << "Error: Invalid dimensions in binary file (h: " << h << ", w: " << w << ", k: " << k << ")." << endl;
        infile.close();
        exit(EXIT_FAILURE);
    }

    // Allocate storage for Dist and Mask
    vector<float> Dist(h * w);
    vector<float> Mask(k * k);

    // Read Dist and Mask
    infile.read(reinterpret_cast<char *>(Dist.data()), h * w * sizeof(float));
    infile.read(reinterpret_cast<char *>(Mask.data()), k * k * sizeof(float));
    infile.close();

    // Open text file for writing
    ofstream outfile(textFile);
    if (!outfile.is_open())
    {
        cerr << "Error: Could not open " << textFile << " for writing." << endl;
        exit(EXIT_FAILURE);
    }

    // Write dimensions
    outfile << h << " " << w << " " << k << endl;

    // Write Dist matrix
    outfile << fixed << setprecision(4); // Format to 4 decimal places
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            outfile << Dist[i * w + j] << " ";
        }
        outfile << endl;
    }

    // Write Mask matrix
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            outfile << Mask[i * k + j] << " ";
        }
        outfile << endl;
    }

    outfile.close();
    cout << "Binary file \"" << binaryFile << "\" successfully converted to text file \"" << textFile << "\"." << endl;
}
void convertBinaryToText_out(const char *binaryFile, const char *textFile)
{
    ifstream infile(binaryFile, ios::binary);
    if (!infile.is_open())
    {
        cerr << "Error: Could not open " << binaryFile << " for reading." << endl;
        exit(EXIT_FAILURE);
    }

    // Read dimensions from the binary file
    int height, width;
    infile.read(reinterpret_cast<char *>(&height), sizeof(int));
    infile.read(reinterpret_cast<char *>(&width), sizeof(int));

    // Validate dimensions
    if (height <= 0 || width <= 0)
    {
        cerr << "Error: Invalid dimensions in binary file (height: " << height << ", width: " << width << ")." << endl;
        infile.close();
        exit(EXIT_FAILURE);
    }

    // Allocate storage for the result matrix
    vector<float> Result(height * width);

    // Read the result matrix
    infile.read(reinterpret_cast<char *>(Result.data()), height * width * sizeof(float));
    if (infile.fail())
    {
        cerr << "Error: Could not read the complete matrix from the binary file." << endl;
        infile.close();
        exit(EXIT_FAILURE);
    }
    infile.close();

    // Open text file for writing
    ofstream outfile(textFile);
    if (!outfile.is_open())
    {
        cerr << "Error: Could not open " << textFile << " for writing." << endl;
        exit(EXIT_FAILURE);
    }

    // Write all values in Result as space-separated values with precision
    outfile << fixed << setprecision(POINT); // Format to 1 decimal place
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            outfile << Result[i * width + j] << " ";
        }
        outfile << endl;
    }

    outfile.close();
    cout << "Binary file \"" << binaryFile << "\" converted to text file \"" << textFile << "\" successfully." << endl;
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cerr << "Usage: ./binary_to_text <binary_input_file> <text_output_file> <result h> <result w>" << endl;
        return EXIT_FAILURE;
    }

    const char *binaryFile = argv[1];
    const char *textFile = argv[2];
    height = atoi(argv[3]);
    width = atoi(argv[4]);

    convertBinaryToText_out(binaryFile, textFile);
    return EXIT_SUCCESS;
}
