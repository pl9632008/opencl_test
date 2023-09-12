//#include <CL/opencl.h>
#include <opencv2/opencv.hpp>
#include <chrono>

int main(int argc , char * argv[]){


    cv::Mat image = cv::imread("../image.jpg");
    cv::Mat templ = cv::imread("../templ.jpg");
    cv::Mat result;
    cv::Mat draw_img;
    cv::Mat draw_uimg;
    image.copyTo(draw_img);
    image.copyTo(draw_uimg);


    auto start = std::chrono::system_clock::now();
    cv::matchTemplate(image,templ,result,3);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result,&minVal,&maxVal,&minLoc,&maxLoc);
    cv::rectangle(draw_img, maxLoc ,cv::Point( templ.cols + maxLoc.x , templ.rows + maxLoc.y ),cv::Scalar(0,255,0) );
    auto end  = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::milliseconds >(end - start).count();
    std::cout<<"mat passing time = " << time <<" ms "<<std::endl;
    cv::imwrite("../result.jpg",draw_img);


    cv::UMat uimage;
    cv::UMat utempl;
    cv::UMat uresult;
    image.copyTo(uimage);
    templ.copyTo(utempl);
    auto start2 = std::chrono::system_clock::now();
    cv::matchTemplate(uimage,utempl,uresult,3);
    double uminVal, umaxVal;
    cv::Point uminLoc, umaxLoc;
    cv::minMaxLoc(uresult,&uminVal,&umaxVal,&uminLoc,&umaxLoc);
    cv::rectangle(draw_uimg, umaxLoc ,cv::Point( utempl.cols + umaxLoc.x , utempl.rows + umaxLoc.y ),cv::Scalar(0,0,255) );
    auto end2  = std::chrono::system_clock::now();
    auto time2 = std::chrono::duration_cast< std::chrono::milliseconds >(end2 - start2).count();
    std::cout<<"umat passing time = " << time2<<" ms "<<std::endl;
    cv::imwrite("../result2.jpg",draw_uimg);



}
