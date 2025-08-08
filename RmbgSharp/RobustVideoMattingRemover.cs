using System.Drawing;
using System.Drawing.Imaging;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace RmbgSharp
{
    /// <summary>
    /// Background remover based on the Robust Video Matting model.
    /// This implementation supports sequential frames by keeping
    /// the recurrent states between calls. For single images the
    /// states are reset for each frame.
    /// </summary>
    public class RobustVideoMattingRemover
    {
        private readonly InferenceSession _session;
        private DenseTensor<float> _r1;
        private DenseTensor<float> _r2;
        private DenseTensor<float> _r3;
        private DenseTensor<float> _r4;

        /// <summary>
        /// Create a new RobustVideoMattingRemover.
        /// </summary>
        /// <param name="modelPath">Path to the ONNX RVM model.</param>
        /// <param name="useFP16">Enable extended optimisations for FP16 models.</param>
        /// <param name="useGPU">Try to run the model on DirectML.</param>
        public RobustVideoMattingRemover(string modelPath, bool useFP16, bool useGPU)
        {
            SessionOptions options = new SessionOptions();
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            if (useFP16)
            {
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;
            }
            if (useGPU)
            {
                try { options.AppendExecutionProvider_DML(); } catch { }
            }
            _session = new InferenceSession(modelPath, options);
            ResetState();
        }

        /// <summary>
        /// Reset internal recurrent states. Call this before processing
        /// a new clip when temporal information should not leak.
        /// </summary>
        public void ResetState()
        {
            _r1 = new DenseTensor<float>(new[] { 1, 16, 1, 1 });
            _r2 = new DenseTensor<float>(new[] { 1, 20, 1, 1 });
            _r3 = new DenseTensor<float>(new[] { 1, 40, 1, 1 });
            _r4 = new DenseTensor<float>(new[] { 1, 64, 1, 1 });
        }

        /// <summary>
        /// Remove the background from a frame using the RVM model.
        /// </summary>
        /// <param name="bitmap">Frame to process.</param>
        /// <returns>Bitmap with alpha channel representing foreground.</returns>
        public Bitmap RemoveBackground(Bitmap bitmap)
        {
            return RemoveBackground(bitmap, 0.25f);
        }

        /// <summary>
        /// Remove the background from a frame using the RVM model with a
        /// custom downsample ratio.
        /// </summary>
        /// <param name="bitmap">Frame to process.</param>
        /// <param name="downsampleRatio">Downsample ratio used by the model.</param>
        /// <returns>Bitmap with alpha channel representing foreground.</returns>
        public Bitmap RemoveBackground(Bitmap bitmap, float downsampleRatio)
        {
            float[,,] image = LoadAndPreprocessImage(bitmap, out int width, out int height);
            DenseTensor<float> src = new DenseTensor<float>(new[] { 1, 3, height, width });
            for (int c = 0; c < 3; c++)
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        src[0, c, y, x] = image[c, y, x];
                    }
                }
            }

            DenseTensor<float> ds = new DenseTensor<float>(new[] { 1 });
            ds[0] = downsampleRatio;

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("src", src),
                NamedOnnxValue.CreateFromTensor("r1i", _r1),
                NamedOnnxValue.CreateFromTensor("r2i", _r2),
                NamedOnnxValue.CreateFromTensor("r3i", _r3),
                NamedOnnxValue.CreateFromTensor("r4i", _r4),
                NamedOnnxValue.CreateFromTensor("downsample_ratio", ds)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

            Tensor<float> fgr = results.First(r => r.Name == "fgr").AsTensor<float>();
            Tensor<float> pha = results.First(r => r.Name == "pha").AsTensor<float>();
            _r1 = (DenseTensor<float>)results.First(r => r.Name == "r1o").AsTensor<float>();
            _r2 = (DenseTensor<float>)results.First(r => r.Name == "r2o").AsTensor<float>();
            _r3 = (DenseTensor<float>)results.First(r => r.Name == "r3o").AsTensor<float>();
            _r4 = (DenseTensor<float>)results.First(r => r.Name == "r4o").AsTensor<float>();

            return ComposeFrame(fgr, pha);
        }

        // Convert network output to a bitmap with alpha channel.
        private Bitmap ComposeFrame(Tensor<float> fgr, Tensor<float> pha)
        {
            int height = fgr.Dimensions[2];
            int width = fgr.Dimensions[3];
            Bitmap result = new Bitmap(width, height);

            BitmapData data = result.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                byte* ptr = (byte*)data.Scan0;
                int stride = data.Stride;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int offset = y * stride + x * 4;
                        float r = fgr[0, 0, y, x];
                        float g = fgr[0, 1, y, x];
                        float b = fgr[0, 2, y, x];
                        float a = pha[0, 0, y, x];

                        ptr[offset] = (byte)(b * 255f);
                        ptr[offset + 1] = (byte)(g * 255f);
                        ptr[offset + 2] = (byte)(r * 255f);
                        ptr[offset + 3] = (byte)(a * 255f);
                    }
                }
            }
            result.UnlockBits(data);
            return result;
        }

        // Preprocess the bitmap by converting pixels to float values in range [0,1].
        private float[,,] LoadAndPreprocessImage(Bitmap bitmap, out int width, out int height)
        {
            width = bitmap.Width;
            height = bitmap.Height;
            float[,,] data = new float[3, height, width];
            BitmapData bmpData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            IntPtr ptr = bmpData.Scan0;
            int bytesPerPixel = Image.GetPixelFormatSize(bmpData.PixelFormat) / 8;
            int stride = bmpData.Stride;

            unsafe
            {
                byte* dataPtr = (byte*)ptr;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int offset = y * stride + x * bytesPerPixel;
                        byte b = dataPtr[offset];
                        byte g = dataPtr[offset + 1];
                        byte r = dataPtr[offset + 2];
                        data[0, y, x] = r / 255f;
                        data[1, y, x] = g / 255f;
                        data[2, y, x] = b / 255f;
                    }
                }
            }

            bitmap.UnlockBits(bmpData);
            return data;
        }
    }
}
