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
int** answerImage;
int numChunks;
double timing = 0;
double dtime_static = 0;
double dtime_dyn = 0;
double avg = 0;
int trialnum = 10;
int NUM_OF_THREADS = 2;
std::vector<std::string> message;
/* ****************Change and add functions below ***************** */
void clearoutput() {
    //clear image
    for (int i = 0; i < image_height; i++)
    {
        for (int j = 0; j < image_width; j++) {
            outputImage[i][j] = 0;
        }
    }
}
void robert(int startx, int starty, int width, int height) {

    int maskX[2][2];
    int maskY[2][2];
    int grad_x;
    int grad_y;
    int grad;
    /* 2x2 Roberts cross mask for X Dimension. */
    maskX[0][0] = +1; maskX[0][1] = 0;
    maskX[1][0] = 0; maskX[1][1] = -1;
    /* 2x2 Roberts cross mask for Y Dimension. */
    maskY[0][0] = 0; maskY[0][1] = +1;
    maskY[1][0] = -1; maskY[1][1] = 0;
    for (int x = startx; x < height; ++x) {
        for (int y = starty; y < width; ++y) {
            grad_x = 0;
            grad_y = 0;
            /* For handling image boundaries */
            if (x == (image_height - 1) || y == (image_width - 1))
                grad = 0;
            else {
                /* Gradient calculation in X Dimension */
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j <= 1; j++) {
                        grad_x += (inputImage[x + i][y + j] * maskX[i][j]);
                    }
                }
                /* Gradient calculation in Y Dimension */
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j <= 1; j++) {
                        grad_y += (inputImage[x + i][y + j] * maskY[i][j]);
                    }
                }
                /* Gradient magnitude */
                grad = (int)sqrt((grad_x * grad_x) + (grad_y * grad_y));
            }
            outputImage[x][y] = grad <= 255 ? grad : 255;

        }
        //Pmtx.lock();
        //std::cout << "x: " << x << std::endl;
        //Pmtx.unlock();
    }
    //std::cout << "x: " << startx << " y:" << starty << "height" << height << "width"  << width << std::endl;
}
void compute_roberts_static() {
    int chunkSize = image_height / numChunks;
    //int threadnum;
    int maskX[2][2];
    int maskY[2][2];
    int grad_x;
    int grad_y;
    int grad;
    std::string temp;
    /* 2x2 Roberts cross mask for X Dimension. */
    maskX[0][0] = +1; maskX[0][1] = 0;
    maskX[1][0] = 0; maskX[1][1] = -1;
    /* 2x2 Roberts cross mask for Y Dimension. */
    maskY[0][0] = 0; maskY[0][1] = +1;
    maskY[1][0] = -1; maskY[1][1] = 0;

    #pragma omp parallel for schedule(static, chunkSize) shared(outputImage,message, chunkSize) private( grad, grad_x, grad_y)
    for (int x = 0; x < image_height; ++x) {
        //threadnum = omp_get_thread_num();
        /*if (x % chunkSize == 0) {
           //    std::cout << "Start Thread " << std::to_string(threadnum) << " Start point: " << std::to_string(x) << std::endl;
            temp = ("Start Thread " + std::to_string(threadnum) + " Start point: " + std::to_string(x) + "\n");
            #pragma omp critical
            message += temp;
        }*/
        for (int y = 0; y < image_width; ++y) {
            grad_x = 0;
            grad_y = 0;
            /* For handling image boundaries */
            if (x == (image_height - 1) || y == (image_width - 1))
                grad = 0;
            else {
                /* Gradient calculation in X Dimension */
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j <= 1; j++) {
                        grad_x += (inputImage[x + i][y + j] * maskX[i][j]);
                    }
                }
                /* Gradient calculation in Y Dimension */
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j <= 1; j++) {
                        grad_y += (inputImage[x + i][y + j] * maskY[i][j]);
                    }
                }
                /* Gradient magnitude */
                grad = (int)sqrt((grad_x * grad_x) + (grad_y * grad_y));
            }
            //#pragma omp critical
            outputImage[x][y] = grad <= 255 ? grad : 255;
        }
        /*
        temp = (std::to_string(threadnum) + " x: " + std::to_string(x) + "\n");

        #pragma omp critical
        message +=  temp;
        /
        if (x % chunkSize == 0) {
            temp = ("End Thread " + std::to_string(threadnum) + "\n");
            #pragma omp critical
            message += temp;

        }
        */
    }
}
void compute_roberts_dynamic()
{
    int chunkSize = image_height / numChunks;
    //int threadnum;

    std::string temp;
    int maskX[2][2];
    int maskY[2][2];
    int grad_x;
    int grad_y;
    int grad;
    /* 2x2 Roberts cross mask for X Dimension. */
    maskX[0][0] = +1; maskX[0][1] = 0;
    maskX[1][0] = 0; maskX[1][1] = -1;
    /* 2x2 Roberts cross mask for Y Dimension. */
    maskY[0][0] = 0; maskY[0][1] = +1;
    maskY[1][0] = -1; maskY[1][1] = 0;

    #pragma omp parallel for schedule(dynamic, chunkSize) shared(outputImage,message, chunkSize) private(grad, grad_x, grad_y)
    for (int x = 0; x < image_height; ++x) {
        /*
        threadnum = omp_get_thread_num();
        if (x % chunkSize == 0) {
            temp = ("Start Thread " + std::to_string(threadnum) + " Start point: " + std::to_string(x) + "\n");
            #pragma omp critical  
            message += "Start Thread " + std::to_string(threadnum) + " Start point: " + std::to_string(x) + "\n";
        }
        */
        for (int y = 0; y < image_width; ++y) {
            grad_x = 0;
            grad_y = 0;
            /* For handling image boundaries */
            if (x == (image_height - 1) || y == (image_width - 1))
                grad = 0;
            else {
                /* Gradient calculation in X Dimension */
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j <= 1; j++) {
                        grad_x += (inputImage[x + i][y + j] * maskX[i][j]);
                    }
                }
                /* Gradient calculation in Y Dimension */
                for (int i = 0; i <= 1; i++) {
                    for (int j = 0; j <= 1; j++) {
                        grad_y += (inputImage[x + i][y + j] * maskY[i][j]);
                    }
                }
                /* Gradient magnitude */
                grad = (int)sqrt((grad_x * grad_x) + (grad_y * grad_y));
            }
            
            //#pragma omp critical
            outputImage[x][y] = grad <= 255 ? grad : 255;
           
        }
        /*
        if (x % chunkSize == 0) {
            temp = ("End Thread " + std::to_string(threadnum) + "\n");
            #pragma omp critical  
            message += temp;
        }
        */
    }
}
/* **************** Change the function below if you need to ***************** */

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cout << "ERROR: Incorrect number of arguments. Format is: <# of chunks> <a1/a2>" << std::endl;
        return 0;
    }
    numChunks = std::atoi(argv[1]);
    std::ofstream data;
    if (argc > 1 && std::string(argv[1]) == "clear") {
        data.open("data.csv", std::ofstream::out | std::ofstream::trunc);
        data.close();
    }
    else {

        for (int i = 1; i <= 6; i++)
        {

            data.open("data.csv", std::ofstream::out | std::ofstream::app);
            if (data.is_open()) { // use is_open();
                data << "Chunk#" << numChunks << " tc" + std::to_string(i) + ".pgm\n";
                data.close();
            }
            else {
                std::cout << "error";
            }
            std::ifstream afile("Lab1_Files_export/tc" + std::to_string(i) + "_solution.pgm");
            std::ifstream file("Lab1_Files_export/tc" + std::to_string(i) + ".pgm");
            numChunks = std::atoi(argv[1]);
            std::cout << "Detect edges in " << argv[1] << " using OpenMP threads\n" << std::endl;

            /* ******Reading image into 2-D array below******** */

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
            std::string workString2;
            while (std::getline(afile, workString2))
            {
                if (workString2.at(0) != '#') {
                    if (workString2.at(1) != '2') {
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
            while (std::getline(afile, workString2))
            {
                if (workString2.at(0) != '#') {
                    std::stringstream stream(workString2);
                    int n;
                    stream >> n;
                    stream >> n;
                    break;
                }
                else {
                    continue;
                }
            }
            inputImage = new int* [image_height];
            outputImage = new int* [image_height];
            answerImage = new int* [image_height];;
            for (int i = 0; i < image_height; ++i) {
                inputImage[i] = new int[image_width];
                outputImage[i] = new int[image_width];
                answerImage[i] = new int[image_width];;
            }

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
            /* Check answer image max shades */
            while (std::getline(afile, workString2))
            {
                if (workString2.at(0) != '#') {
                    std::stringstream stream(workString2);
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

            /* Fill input answer matrix */
            for (int i = 0; i < image_height; i++)
            {
                if (std::getline(afile, workString2) && workString2.at(0) != '#') {
                    std::stringstream stream(workString2);
                    for (int j = 0; j < image_width; j++) {
                        if (!stream)
                            break;
                        stream >> pixel_val;
                        answerImage[i][j] = pixel_val;
                    }
                }
                else {
                    continue;
                }
            }

            /************ Call functions to process image *********/

            std::string opt = argv[2];

            for (int i = 0; i < trialnum; i++)
            {
                clearoutput();
                double dtime_static = omp_get_wtime();
                compute_roberts_static();
                dtime_static = omp_get_wtime() - dtime_static;
                timing += dtime_static;
            }
            dtime_static = timing / trialnum;
            clearoutput();
            for (int i = 0; i < trialnum; i++)
            {
                clearoutput();
                double dtime_dyn = omp_get_wtime();
                compute_roberts_dynamic();
                dtime_dyn = omp_get_wtime() - dtime_dyn;
                timing += dtime_dyn;
                
            }
            dtime_dyn = timing / trialnum;
        
            /* ********Start writing output to your file************ */
            std::ofstream ofile("output/tc" + std::to_string(i) + "o.pgm");
            if (ofile.is_open())
            {
                ofile << "P2" << "\n" << image_width << " " << image_height << "\n" << image_maxShades << "\n";
                for (int i = 0; i < image_height; i++)
                {
                    for (int j = 0; j < image_width; j++) {
                        ofile << outputImage[i][j] << " ";
                    }
                    ofile << "\n";
                }
            }
            else {
                std::cout << "ERROR: Could not open output file " << argv[2] << std::endl;
                return 0;
            }

            //std::cout << "write\n";
            //for (int i = 0; i < message.size(); i++) {
                //std::cout << message.at(i) << std::endl;
            //}
            
            data.open("data.csv", std::ofstream::out | std::ofstream::app);
            if (data.is_open()) { // use is_open();
                data << " " << "," << dtime_static << "," << dtime_dyn;
                data << "\n";
                data.close();
            }
            else {
                std::cout << "error";
            }
            bool fail = false;
            /*
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 10; j++) {
                    std::cout << "out: " << outputImage[i][j] << "answer: " << answerImage[i][j] << std::endl;
                }
            }
            */
            for (int i = 0; i < image_height; i++)
            {
                for (int j = 0; j < image_width; j++) {
                    if (outputImage[i][j] != answerImage[i][j]) {
                        fail = true;
                    }
                }
            }
            if (fail == false) {
                std::cout << "sucess" << std::endl;
            }
            else {
                std::cout << "fail" << std::endl;
            }
            for (int i = 0; i < image_height; ++i) {
                delete[] inputImage[i];
                delete[] outputImage[i];
                delete[] answerImage[i];
            }
            delete[] inputImage;
            delete[] outputImage;
            delete[] answerImage;
        }
    }
    return 0;
}