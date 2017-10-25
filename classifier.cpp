#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "Header.h"
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

using namespace cv;
using namespace cv::ml;
using namespace std;


template<typename T = uchar, typename U = float>
void transform_to_svm_training_data(cv::Mat &input)
{
  if(input.isContinuous()){
    input = input.reshape(1, 1);
    input.convertTo(input, cv::DataType<U>().depth);
    return;
  }
  cv::Mat output(1, input.total() * input.channels(), cv::DataType<U>().depth);
  auto output_ptr = output.ptr<U>(0);
  OCV::for_each_channels<T>(input, [&](T a)
                            {
                              *output_ptr = a;
                              ++output_ptr;
                            });
  input = output;
}


int main()
{
  cv::Mat trainingImages;
  std::vector<int> trainingLabels;
  std::stringstream ss;

  int count = 7000;

  int k = 0;
  int s = 0;
  ifstream file ( "/Users/mak/Documents/Programming/Programs/CV/CV/gt_train.csv" );
  string value;
  file >> value;
  int cl;
  std::string::size_type sz;
  while (k < 25000) {
    //    cout << k << endl;
    std::stringstream ss;
    ss << "/Users/mak/Documents/Programming/Programs/CV/CV/train/" << std::setw(6) << std::setfill('0') << s << ".png";
    //    cout << ss.str() << endl;
    cv::Mat img = cv::imread(ss.str(), 0);
    transform_to_svm_training_data(img);
    trainingImages.push_back(img);
    file >> value;
    cl = stoi (string(value, 11, value.length()), &sz);
    trainingLabels.emplace_back(cl);
    s++;
    k++;
  }

  Ptr<SVM> svm = SVM::create();
  svm->setType(SVM::C_SVC);
  svm->setKernel(SVM::LINEAR);
  svm->train(trainingImages, ROW_SAMPLE, trainingLabels);


  k = 0;
  s = 0;
  int arr[count];
  while (k < count) {
    //    cout << k << endl;
    std::stringstream ss;
    ss << "/Users/mak/Documents/Programming/Programs/CV/CV/test/" << std::setw(6) << std::setfill('0') << s << ".png";
    //    cout << ss.str() << endl;
    cv::Mat img = cv::imread(ss.str(), 0);
    transform_to_svm_training_data(img);
    int response = svm->predict(img);
    cout << response << endl;
    arr[k] = response;
    s++;
    k++;
  }
  k = 0;
  int wow = 0;
  ifstream file2 ( "/Users/mak/Documents/Programming/Programs/CV/CV/gt_test.csv" );
  string value2;
  file2 >> value2;
  int cl2;
  std::string::size_type sz2;
  while (k < count) {
    file2 >> value2;
    //    cout << value2 << endl;
    cl2 = stoi (string(value2, 11, value2.length()), &sz2);
    if (arr[k] != cl2) {
      cout <<"false"<< arr[k] << " " << cl2 << endl;

    }
    else{
      wow++;
    }
    //    cout <<"true"<< arr[k] << endl;
    k++;
  }

  //  std::stringstream ss3("/Users/mak/Documents/Programming/Programs/CV/CV/000011.png");
  //  cv::Mat img3 = cv::imread(ss3.str(), 0);
  //  transform_to_svm_training_data(img3);
  //
  //  int response = svm->predict(img3);
  //  cout << response << endl;






  cout << "Wow" << wow << endl;

  //  cv::Mat classes;
  //  cv::Mat trainingData = trainingImages;
  //  cv::Mat(trainingLabels).copyTo(classes);
  //  trainingData.convertTo(trainingData, CV_32FC1);
  //  cv::FileStorage fs("SVM.xml", cv::FileStorage::WRITE);
  //  fs << "TrainingData" << trainingData;
  //  fs << "classes" << classes;
  return 0;
}


