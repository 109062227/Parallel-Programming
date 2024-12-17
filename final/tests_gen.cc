#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cmath>
#include <cfloat> // For FLT_MAX

using namespace std;

// Function to generate a random float value within a safe range
float randomFloat()
{
    // float maxVal = 100.0f;
    // float maxVal = sqrt(FLT_MAX) - 1e3f; // square root of float
    // return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * maxVal;
    // range from -1 to 1
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
}
// Main function to generate binary file
int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cerr << "Usage: " << argv[0] << " <height> <width> <mask_size> <output_file>" << endl;
        return EXIT_FAILURE;
    }

    // Parse dimensions and output file name from command line
    int h = atoi(argv[1]);                // Height
    int w = atoi(argv[2]);                // Width
    int k = atoi(argv[3]);                // Mask size
    const char *outputFileName = argv[4]; // Output file name

    // Validate dimensions
    if (h <= 0 || w <= 0 || k <= 0 || k > min(h, w))
    {
        cerr << "Error: Invalid dimensions. Ensure h > 0, w > 0, k > 0, and k <= min(h, w)." << endl;
        return EXIT_FAILURE;
    }

    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Calculate total sizes
    int sizeDist = h * w; // h * w elements for Dist
    int sizeMask = k * k; // k * k elements for Mask

    // Create vectors to hold random values
    vector<float> Dist(sizeDist);
    vector<float> Mask(sizeMask);

    // Fill Dist with random values
    for (int i = 0; i < sizeDist; ++i)
    {
        Dist[i] = randomFloat();
    }

    // Fill Mask with random values
    for (int i = 0; i < sizeMask; ++i)
    {
        Mask[i] = randomFloat();
    }

    // Open binary file for writing
    ofstream outfile(outputFileName, ios::binary);
    if (!outfile.is_open())
    {
        cerr << "Error: Could not open " << outputFileName << " for writing." << endl;
        return EXIT_FAILURE;
    }

    // Write dimensions
    outfile.write(reinterpret_cast<char *>(&h), sizeof(int));
    outfile.write(reinterpret_cast<char *>(&w), sizeof(int));
    outfile.write(reinterpret_cast<char *>(&k), sizeof(int));

    // Write Dist values
    outfile.write(reinterpret_cast<char *>(Dist.data()), sizeDist * sizeof(float));

    // Write Mask values
    outfile.write(reinterpret_cast<char *>(Mask.data()), sizeMask * sizeof(float));

    // Close the file
    outfile.close();

    cout << "Binary file \"" << outputFileName << "\" generated successfully." << endl;

    return EXIT_SUCCESS;
}
