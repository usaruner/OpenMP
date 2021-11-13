#include <omp.h>
#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>

/* Global variables, Look at their usage in main() */
int image_height;
int image_width;
int image_maxShades;
int** inputImage;
int** outputImage;
int numChunks;
double timing = 0;
std::string message = "";
int hist[256];
int answer[256];
int doneChunk = 0;
int chunkSize;
/* ****************Change and add functions below ***************** */
bool boolean() {
    bool stm;
    int temp;
    #pragma omp atomic read
    temp = doneChunk;
    #pragma omp critical
    stm = (chunkSize * temp < image_height);
    //std::cout << "bool" << stm << " sz " << temp <<  std::endl;
    return stm;
}
void computehist() {
    int limit = 0;
    int threadnum = 0;
    #pragma omp parallel shared(hist,inputImage,doneChunk) private(limit, threadnum)
    {
        

        while (boolean()) {
            
            #pragma omp single 
            {
                #pragma omp task
                {
                    if (chunkSize * (doneChunk + 1) > image_height) {
                        limit = image_height;
                    }
                    else {
                        limit = chunkSize * (doneChunk + 1);
                    }
                    threadnum = omp_get_thread_num();
                    std::cout << threadnum << " place: " << doneChunk  << " sum: " << limit << std::endl;
                    for (int x = chunkSize*doneChunk; x < limit; ++x) {
                        for (int y = 0; y < image_width; ++y) {
                            //std::cout << "place: " << inputImage[x][y];
                            //std::cout << "place: " << inputImage[x][y];
                            #pragma omp atomic update
                            hist[inputImage[x][y]]++;
                        }
                    }
                    #pragma omp atomic update
                    doneChunk++;
                }
            }
        }

        #pragma omp taskwait
    }
    /*
    for (int x = chunkSize * doneChunk; x < image_height; ++x) {
        for (int y = 0; y < image_width; ++y) {
            hist[inputImage[x][y]]++;
        }
    }*/
}
/* **************** Change the function below if you need to ***************** */

int main(int argc, char* argv[])
{
    std::cout << "START" << std::endl;
    
    if (argc != 4)
    {
        std::cout << "ERROR: Incorrect number of arguments. Format is: <Input image filename>  <# of chunks> <Output image filename>" << std::endl;
        return 0;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open())
    {
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        return 0;
    }
    numChunks = std::atoi(argv[2]);

    std::cout << "Detect edges in " << argv[1] << " using OpenMP threads\n" << std::endl;

    /* ******Reading image into 2-D array below******** */
    for (int i = 0; i < 256; i++) {
        hist[i] = 0;
        answer[i] = 0;
    }

    std::string workString;
    /* Remove comments '#' and check image format */
    while (std::getline(file, workString))
    {
        if (workString.at(0) != '#') {
            if (workString.at(1) != '2') {
                std::cout << "Input image is not a valid PGM image" << std::endl;
                return 0;
            }
            else {
                break;
            }
        }
        else {
            continue;
        }
    }
    /* Check image size */
    while (std::getline(file, workString))
    {
        if (workString.at(0) != '#') {
            std::stringstream stream(workString);
            int n;
            stream >> n;
            image_width = n;
            stream >> n;
            image_height = n;
            break;
        }
        else {
            continue;
        }
    }

    inputImage = new int* [image_height];
    outputImage = new int* [image_height];
    for (int i = 0; i < image_height; ++i) {
        inputImage[i] = new int[image_width];
        outputImage[i] = new int[image_width];
    }
    chunkSize = image_height / numChunks;
    /* Check image max shades */
    while (std::getline(file, workString))
    {
        if (workString.at(0) != '#') {
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        }
        else {
            continue;
        }
    }
    /* Fill input image matrix */
    int pixel_val;
    for (int i = 0; i < image_height; i++)
    {
        if (std::getline(file, workString) && workString.at(0) != '#') {
            std::stringstream stream(workString);
            for (int j = 0; j < image_width; j++) {
                if (!stream)
                    break;
                stream >> pixel_val;
                inputImage[i][j] = pixel_val;
            }
        }
        else {
            continue;
        }
    }

    /************ Call functions to process image *********/
    std::cout << "START" << std::endl;
    doneChunk = 0;
    computehist();
    std::cout << "DONE" << std::endl;
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            answer[inputImage[j][i]]++;
        }
    }
    std::ofstream ofile(argv[3]);
    if (ofile.is_open()) {
        for (int i = 0; i < 256; i++) {
            ofile << i << " : " << hist[i] << " answer " << " : " << answer[i] << "\n";
        }
    }
    else {
        std::cout << "ERROR: Could not open output file " << argv[3] << std::endl;
        return 0;
    }
    std::cout << "DONE" << std::endl;
    return 0;
}