extern crate opencv;
use opencv::core as cv;
use opencv::sys::types as cvtypes;
use opencv::{imgproc, objdetect, highgui};

fn run() -> Result<(), String> {
    let mut classifier =
        try!(objdetect::CascadeClassifier::for_file("haarcascade_frontalface_alt.xml"));

    let mut capture = try!(highgui::VideoCapture::for_device(-1));
    try!(highgui::namedWindow("facedetect", 0));
    loop {
        if try!(capture.grab()) {
            let mut image0 = cv::mat();
            try!(capture.retrieve(&mut image0, 0));
            let mut image1 = cv::mat();
            try!(imgproc::cvtColor(&image0, &mut image1, imgproc::CV_BGR2GRAY, 0));
            let mut image2 = cv::mat();
            try!(imgproc::equalizeHist(&image1, &mut image2));

            let mut rects = cvtypes::VectorOfRect::new();
            try!(classifier.detectMultiScale(&image2, &mut rects, 1.1, 3, 0,
                 cv::Size { width:0, height:0 }, cv::Size { width:0, height:0 }));

            for rect in rects.into_vec().iter() {
                let src = cv::Point { x: rect.x, y: rect.y };
                let dst = cv::Point { x: rect.x + rect.width, y: rect.y + rect.height };
                cv::rectangle(&mut image0, src, dst, cv::Scalar { data: [144.0, 48.0, 255.0, 0.0] },
                              1, 8, 0);
            }

            try!(highgui::imshow("facedetect", &image0));
        }

        if try!(highgui::waitKey(10)) == 27 { break }
    }
    try!(highgui::destroyWindow("facedetect"));
    try!(capture.release());
    Ok(())
}

fn main() {
    run().unwrap()
}
