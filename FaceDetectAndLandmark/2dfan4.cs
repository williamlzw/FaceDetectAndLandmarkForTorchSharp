using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using TorchSharp;
using static TorchSharp.torch;


namespace FaceLandmark
{
    public class LandmarkResult
    {
        /// <summary>
        /// 坐标数组，68个
        /// </summary>
        public List<Point> Points { get; set; } = new List<Point>();
        /// <summary>
        /// 分数
        /// </summary>
        public float Score { get; set; }
    }

    public class FaceLandmark_2dfan4b
    {
        private readonly InferenceSession _session;
        public FaceLandmark_2dfan4b(string model_file)
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
        /// <param name="img">[C, H, W]</param>
        /// <param name="bounding_box">[4] x1,y1,x2,y2</param>
        /// <returns></returns>
        public LandmarkResult GetLandmark(torch.Tensor img, torch.Tensor bounding_box)
        {
            var scale = 195f / torch.subtract(bounding_box[TensorIndex.Slice(2, null)], bounding_box[TensorIndex.Slice(null, 2)]).max();
            var translation = (256f - torch.add(bounding_box[TensorIndex.Slice(2, null)], bounding_box[TensorIndex.Slice(null, 2)]) * scale) * 0.5f;
            var(cropVisionFrame, affineMatrix) = WarpFaceByTranslation(img, translation, scale, new torch.Size(new int[] { 256, 256 }));
            cropVisionFrame = cropVisionFrame / 255f;
            var runOptions = new RunOptions();
            var ortValue = OrtValue.CreateTensorValueFromMemory(cropVisionFrame.data<float>().ToArray(), new long[] { 1, 3, 256, 256 });
            var inputs = new Dictionary<string, OrtValue>
             {
                 { "input", ortValue }
             };
            var results = _session.Run(runOptions, inputs, _session.OutputNames);
            var faceLandmark68 = results[0].GetTensorDataAsSpan<float>().ToArray();
            var faceHeatmap = results[1].GetTensorDataAsSpan<float>().ToArray();
            var faceLandmark68Tensor = torch.tensor(faceLandmark68).reshape(1, 68, 3);
            var faceHeatmapTensor = torch.tensor(faceHeatmap).reshape(1, 68, 64, 64);
            faceLandmark68Tensor = faceLandmark68Tensor[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(null, 2)] / 64 * 256;
            faceLandmark68Tensor = TransformTorch(faceLandmark68Tensor[0], affineMatrix);
            var faceLandmark68Score = torch.amax(faceHeatmapTensor, new long[] { 2, 3 });
            faceLandmark68Score = torch.mean(faceLandmark68Score);
            faceLandmark68Tensor = faceLandmark68Tensor.to(torch.int32);
            return ConvertResult(faceLandmark68Tensor, faceLandmark68Score);
        }

        private static LandmarkResult ConvertResult(torch.Tensor faceLandmark68Tensor, torch.Tensor faceLandmark68Score)
        {
            LandmarkResult result = new LandmarkResult();
            result.Score = faceLandmark68Score.item<float>();
            foreach (var i in Enumerable.Range(0, (int)faceLandmark68Tensor.shape[0]))
            {
                int x = faceLandmark68Tensor[i][0].item<int>();
                int y = faceLandmark68Tensor[i][1].item<int>();
                result.Points.Add(new Point(x, y));
            }
            return result;
        }

        private static torch.Tensor ConvertAffinematrixToHomography(torch.Tensor M)
        {
            var H =torch.nn.functional.pad(M, new long[] { 0, 0, 0, 1 }, PaddingModes.Constant, 0);
            H[TensorIndex.Ellipsis, -1, -1] += 1.0;
            return H;
        }
        private static torch.Tensor NormalTransformPixel(int height, int width)
        {
            var tr_mat = torch.tensor(new float[] { 1f, 0f, -1f, 0f, 1f, -1f, 0f, 0f, 1f }).reshape(3,3);
            float width_denom, height_denom;
            if (width == 1)
            {
                width_denom = (float)1e-14;
            }
            else
            {
                width_denom = width - 1;
            }
            if (height == 1)
            {
                height_denom = (float)1e-14;
            }
            else
            {
                height_denom = height - 1;
            }
            tr_mat[0,0] = tr_mat[0,0] * 2.0f / width_denom;
            tr_mat[1,1] = tr_mat[1,1] * 2.0f / height_denom;
            return tr_mat.unsqueeze(0);
        }
        private static torch.Tensor NormalizeHomography(torch.Tensor dstPixTransSrcPix, torch.Size dsizeSrc, torch.Size dsizeDst)
        {
            var srcH = dsizeSrc[0];
            var srcW = dsizeSrc[1];
            var dstH = dsizeDst[0];
            var dstW = dsizeDst[1];
            var srcNormTransSrcPix = NormalTransformPixel((int)srcH, (int)srcW);
            var dstNormTransDstPix = NormalTransformPixel((int)dstH, (int)dstW);
            var srcPixTransSrcNorm = torch.linalg.inv(srcNormTransSrcPix);
            var dstNormTransSrcNorm = dstNormTransDstPix.matmul(dstPixTransSrcPix.matmul(srcPixTransSrcNorm));
            return dstNormTransSrcNorm;
        }

        /// <summary>
        /// cv2.invertAffineTransform
        /// </summary>
        /// <param name="M">[1,2,3]</param>
        /// <returns></returns>
        private static torch.Tensor InverseAffineTransformTorch(torch.Tensor M)
        {
            var invMat = torch.zeros_like(M);
            var div1 = M[TensorIndex.Colon, 0, 0] * M[TensorIndex.Colon, 1, 1] - M[TensorIndex.Colon, 0, 1] * M[TensorIndex.Colon, 1, 0];
            invMat[TensorIndex.Colon, 0, 0] = M[TensorIndex.Colon, 1, 1] / div1;
            invMat[TensorIndex.Colon, 0, 1] = -M[TensorIndex.Colon, 0, 1] / div1;
            invMat[TensorIndex.Colon, 0, 2] = -(M[TensorIndex.Colon, 0, 2] * M[TensorIndex.Colon, 1, 1] - M[TensorIndex.Colon, 0, 1] * M[TensorIndex.Colon, 1, 2]) / div1;
            var div2 = M[TensorIndex.Colon, 0, 1] * M[TensorIndex.Colon, 1, 0] - M[TensorIndex.Colon, 0, 0] * M[TensorIndex.Colon, 1, 1];
            invMat[TensorIndex.Colon, 1, 0] = M[TensorIndex.Colon, 1, 0] / div2;
            invMat[TensorIndex.Colon, 1, 1] = -M[TensorIndex.Colon, 0, 0] / div2;
            invMat[TensorIndex.Colon, 1, 2] = -(M[TensorIndex.Colon, 0, 2] * M[TensorIndex.Colon, 1, 0] - M[TensorIndex.Colon, 0, 0] * M[TensorIndex.Colon, 1, 2]) / div2;
            return invMat;
        }

        /// <summary>
        /// cv2.warpAffine
        /// </summary>
        /// <param name="src">[C,H,W]</param>
        /// <param name="M">[1,H,W]</param>
        private static torch.Tensor WarpAffineTorch(torch.Tensor src, torch.Tensor M, torch.Size dstSize)
        {
            int H = (int)src.shape[2];
            int W = (int)src.shape[3];
            var matric3x3 = ConvertAffinematrixToHomography(M);
            var dstNormTransSrcNorm = NormalizeHomography(matric3x3, new torch.Size(new int[] { H, W }), dstSize);
            var srcNormTransDstNorm = torch.linalg.inv(dstNormTransSrcNorm);
            var grid = torch.nn.functional.affine_grid(srcNormTransDstNorm[TensorIndex.Colon, TensorIndex.Slice(null, 2), TensorIndex.Colon], size: new long[] { 1, 3, 256, 256 }, align_corners:true);
            var warpSrc = torch.nn.functional.grid_sample(src.to(torch.float32), grid, padding_mode:GridSamplePaddingMode.Border, align_corners:true);
            return warpSrc;
        }

        /// <summary>
        /// cv2.transform
        /// </summary>
        /// <param name="points">[N,2]</param>
        /// <param name="affine_matrix">[2,3]</param>
        /// <returns></returns>
        private static torch.Tensor TransformTorch(torch.Tensor points, torch.Tensor affineMatrix)
        {
            points = torch.hstack(new torch.Tensor[] { points, torch.ones(points.shape[0], 1) });
            var eye = torch.eye(3);
            eye[TensorIndex.Slice(null, 2), TensorIndex.Colon] = affineMatrix;
            var transformedTensor = torch.matmul(points, eye.t());
            transformedTensor = transformedTensor[TensorIndex.Colon, TensorIndex.Slice(null, 2)] / transformedTensor[TensorIndex.Colon, TensorIndex.Single(2)].unsqueeze(1);
            return transformedTensor;
        }

        private static (torch.Tensor, torch.Tensor) WarpFaceByTranslation(torch.Tensor src, torch.Tensor translation, torch.Tensor scale, torch.Size dstSize)
        {
            float[] arr = new float[] { scale.item<float>(), 0f, translation[0].item<float>(), 0f, scale.item<float>(), translation[1].item<float>()};
            var M = torch.tensor(arr).reshape(2,3).unsqueeze(0);
            var affineMatrix = InverseAffineTransformTorch(M);
            var warpSrc = WarpAffineTorch(src.unsqueeze(0), M, dstSize);
            return (warpSrc, affineMatrix);
        }
    }
}