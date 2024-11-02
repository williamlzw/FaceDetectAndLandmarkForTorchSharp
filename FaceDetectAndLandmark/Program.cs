using OpenCvSharp;
using FaceLandmark;
using TorchSharp;
using FaceDetect;


namespace FaceDetectAndLandmark
{
    public class Test
    {
        public static Mat DrawLmks(Mat img, LandmarkResult result)
        {
            foreach (var i in Enumerable.Range(0, result.Points.Count()))
            {
                Cv2.Circle(img, result.Points[i], 1, new OpenCvSharp.Scalar(0, 0, 255), 1);
            }
            return img;
        }

        public static Mat DrawBox(Mat img, DetectResult result)
        {
            Cv2.Rectangle(img, result.BoundingBox, new OpenCvSharp.Scalar(255, 0, 0), 1);
            //foreach (var i in Enumerable.Range(0, result.Landmark5.Count()))
            //{
            //    Cv2.Circle(img, result.Landmark5[i], 1, new OpenCvSharp.Scalar(0, 0, 255), 1);
            //}
            return img;
        }

        public static void Main()
        {
            var toolDetect = new FaceDetect_Yolo8nface(".\\data\\yoloface_8n.onnx");
            var toolLandMark = new FaceLandmark_2dfan4b(".\\data\\2dfan4.onnx");

            var imgPath = ".\\data\\6.jpg";
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();
            var imgTensor = torchvision.io.read_image(imgPath);
            // Console.WriteLine(string.Join(",", imgTensor.shape.Select(x => x.ToString()).ToArray()));
            var time0 = DateTime.Now;
            var resultDetect = toolDetect.Detect(imgTensor);
            var time1 = DateTime.Now;
            Console.WriteLine("detect time:" + (time1 - time0).TotalMilliseconds);

            var boundingBox = torch.from_array(new int[]{resultDetect.BoundingBox.Left,
                resultDetect.BoundingBox.Top, resultDetect.BoundingBox.Right,
                resultDetect.BoundingBox.Bottom});

            var time2 = DateTime.Now;
            var resultLandmark = toolLandMark.GetLandmark(imgTensor, boundingBox);
            var time3 = DateTime.Now;
            Console.WriteLine("landmark time:" + (time3 - time2).TotalMilliseconds);

            var img = Cv2.ImRead(imgPath);
            var retImg = DrawBox(img, resultDetect);
            retImg = DrawLmks(retImg, resultLandmark);
            Cv2.ImWrite("result.jpg", retImg);
        }
    }  
}
