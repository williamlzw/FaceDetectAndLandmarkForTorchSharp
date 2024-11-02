using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;


namespace FaceDetect
{
    public class DetectResult
    {
        public Rect BoundingBox { get; set; }
        public List<Point> Landmark5 { get; set; } = new List<Point>();
        public float Score { get; set; }
    }
    public class FaceDetect_Yolo8nface
    {
        private readonly InferenceSession _session;
        public FaceDetect_Yolo8nface(string model_file)
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            // 设置为CPU上运行
            options.AppendExecutionProvider_CPU(0);
            _session = new InferenceSession(model_file, options);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="input">[1,C,H,W]</param>
        /// <param name="maxResolution"></param>
        /// <returns></returns>
        private torch.Tensor ResizeProcess(torch.Tensor input, int maxResolution = 640)
        {
            int height = (int)input.shape[2];
            int width = (int)input.shape[3];
            if (height > maxResolution | width > maxResolution)
            {
                float scale = Math.Min((float)maxResolution/ height, (float)maxResolution/ width);
                int newWidth = (int)(width * scale);
                int newHeight = (int)(height * scale);
                return torch.nn.functional.interpolate(input, size: new long[] { newHeight, newWidth }, mode: InterpolationMode.Bilinear, align_corners: false);
            }
            return input;
        }

        private torch.Tensor PrepareData(torch.Tensor data, int detectSize)
        {
            var detectTensor = torch.zeros(new long[] { 1, data.shape[1], detectSize, detectSize }, dtype:ScalarType.Byte);
            detectTensor[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, data.shape[2]), TensorIndex.Slice(null, data.shape[3])] = data;
            detectTensor = (detectTensor.to(torch.float32) - 127.5f) / 128.0f;
            return detectTensor;
        }

        private static DetectResult ConvertResult(torch.Tensor boundingBox, torch.Tensor landmark5, torch.Tensor score)
        {
            var result = new DetectResult();
            var box = boundingBox.data<int>().ToArray();
            result.BoundingBox = new Rect(box[0], box[1],  box[2]-box[0], box[3]-box[1]);
            foreach (var i in Enumerable.Range(0, (int)landmark5.shape[0]))
            {
                int x = landmark5[i][0].item<int>();
                int y = landmark5[i][1].item<int>();
                result.Landmark5.Add(new Point(x, y));
            }
            result.Score = score.item<float>();
            return result;
        }

        public DetectResult Detect(torch.Tensor img, float detectScoreThreold = 0.5f)
        {
            int detectSize = 640;
            var newImg = img.unsqueeze(0);
            var resizeImg = ResizeProcess(newImg, detectSize);
            var ratioHeight = (float)newImg.shape[2] / resizeImg.shape[2];
            var ratioWidth = (float)newImg.shape[3] / resizeImg.shape[3];
            var detectTensor = PrepareData(resizeImg, detectSize);
            var runOptions = new RunOptions();
            var ortValue = OrtValue.CreateTensorValueFromMemory(detectTensor.data<float>().ToArray(), new long[] { 1, 3, detectSize, detectSize });
            var inputs = new Dictionary<string, OrtValue>
             {
                 { "images", ortValue }
             };
            var results = _session.Run(runOptions, inputs, _session.OutputNames);
            var detections = results[0].GetTensorDataAsSpan<float>().ToArray();
            var detectionsTensor = torch.tensor(detections).reshape(20, -1).t();
            var splitResult = torch.split(detectionsTensor, new long[] { 4, 1, 15 }, dim: 1);
            var boundingBoxRaw = splitResult[0];
            var scoreRaw = splitResult[1];
            var faceLandmark5Raw = splitResult[2];
            var keep_indices = torch.where(scoreRaw > detectScoreThreold)[0];
            if (keep_indices.any().item<bool>())
            {
                boundingBoxRaw = boundingBoxRaw[keep_indices];
                scoreRaw = scoreRaw[keep_indices];
                faceLandmark5Raw = faceLandmark5Raw[keep_indices];
                var boundingBoxList = torch.stack(new List<Tensor>
                {
                    (boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(0)] - boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(2)] / 2) * ratioWidth,
                    (boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(1)] - boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(3)] / 2) * ratioHeight,
                    (boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(0)] + boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(2)] / 2) * ratioWidth,
                    (boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(1)] + boundingBoxRaw[TensorIndex.Colon, TensorIndex.Single(3)] / 2) * ratioHeight
                }, dim:1);
                faceLandmark5Raw[TensorIndex.Colon, TensorIndex.Slice(0, null, 3)] = faceLandmark5Raw[TensorIndex.Colon, TensorIndex.Slice(0, null, 3)] * ratioWidth;
                faceLandmark5Raw[TensorIndex.Colon, TensorIndex.Slice(1, null, 3)] = faceLandmark5Raw[TensorIndex.Colon, TensorIndex.Slice(1, null, 3)] * ratioHeight;
                var faceLandmark5List = faceLandmark5Raw.reshape(-1, 5, 3)[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, 2)];
                var scoreList = scoreRaw.ravel();
                var sortIndices = torch.argsort(-scoreList, dim: 0);
                var boundingBoxListSort = boundingBoxList[sortIndices];
                var faceLandmark5ListSort = faceLandmark5List[sortIndices];
                var scoreListSort = scoreList[sortIndices];
                var nmsIndices = torchvision.ops.nms(boundingBoxListSort, scoreListSort, 0.4);
                var boundingBoxListNms = boundingBoxListSort[nmsIndices];
                var faceLandmark5ListNms = faceLandmark5ListSort[nmsIndices];
                var scoreListNms = scoreListSort[nmsIndices];
                return ConvertResult(boundingBoxListNms[0].to(torch.int32), faceLandmark5ListNms[0].to(torch.int32), scoreListNms[0]);
            }
            return new DetectResult();
        }
    }
}
