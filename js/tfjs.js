import * as tf from "@tensorflow/tfjs";

const emnistLabels = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
  'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
  'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q',
  'r', 't'
];


let model;
let drawing = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANVAS_SIZE = 280;
const modelInputSize = 28;
canvas.width = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

ctx.fillStyle = "white";
ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const predictionDiv = document.getElementById("prediction");

export const tfjs = async () => {

  predictionDiv.innerText = "Loading model...";

  try {

    await tf.setBackend("webgl");
    await tf.ready();
  } catch {

    await tf.setBackend("cpu");
    await tf.ready();
  }

  try {

    model = await tf.loadGraphModel("tfjs_model/model.json");
    predictionDiv.innerText = "Model ready. Draw and click Predict.";
    console.log("Model loaded.");
  } catch (error) {

    predictionDiv.innerText = "Failed to load model.";
    console.error("Model loading error:", error);
  }
};

predictBtn.onclick = async () => {
  if (!model) {
    predictionDiv.textContent = "Model not loaded yet, please wait.";
    return;
  }

  try {
   tf.tidy(() => {
      // 1. Create tensor from canvas pixels: shape [height, width, 4]
      let imgTensor = tf.browser.fromPixels(canvas);

      // 2. Convert to grayscale (average RGB channels)
      imgTensor = imgTensor.slice([0, 0, 0], [-1, -1, 3]);
      const grayscaleTensor = imgTensor.mean(2); // shape [height, width]

      // 3. Resize to model input size (28x28)
      const resizedTensor = tf.image.resizeBilinear(
        grayscaleTensor.expandDims(-1),
        [modelInputSize, modelInputSize]
      );

      // 4. Normalize to [0,1] and invert colors (white background → 0, black strokes → 1)
      const normalizedTensor = tf.div(tf.sub(255, resizedTensor), 255);

      // 5. Add batch dimension: shape [1, 28, 28, 1]
      const inputTensor = normalizedTensor.expandDims(0);

      // 6. Predict (output shape [1, 47])
      const outputTensor = model.predict(inputTensor);

      // 7. Extract probabilities outside tidy (to avoid disposal)
      const probs = outputTensor.dataSync();

      // 8. Find max probability index and value
      const maxIdx = probs.reduce(
        (maxIndex, prob, i, arr) => (prob > arr[maxIndex] ? i : maxIndex),
        0
      );

      const maxProb = probs[maxIdx];

      // 9. Map prediction index to character label from emnistLabels
      //    Fallback to index if label missing
      const predictedLabel = emnistLabels?.[maxIdx] ?? maxIdx;

      // 10. Display prediction and confidence
      predictionDiv.innerHTML = `
        <b>Predicted:</b> ${predictedLabel} <br/>
        <b>Confidence:</b> ${(maxProb * 100).toFixed(2)}%
      `;
    });
  } catch (error) {
    predictionDiv.textContent = `Error during prediction: ${error.message}`;
    console.error(error);
  }
};

function getCanvasCoords(e) {

  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;
  return { x, y };
}

canvas.addEventListener("pointerdown", e => {

  if (["mouse", "pen", "touch"].includes(e.pointerType)) {

    drawing = true;
    const { x, y } = getCanvasCoords(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
    e.preventDefault();
  }
});

canvas.addEventListener("pointermove", e => {

  if (drawing) {

    const { x, y } = getCanvasCoords(e);
    ctx.strokeStyle = "black";
    ctx.lineWidth = Math.max(10, Math.min(canvas.width, canvas.height) / 20);
    ctx.lineCap = "round";
    ctx.lineTo(x, y);
    ctx.stroke();
    e.preventDefault();
  }
});

["pointerup", "pointercancel", "pointerleave"].forEach(evt => {

  canvas.addEventListener(evt, () => {

    drawing = false;
  });
});

clearBtn.addEventListener("click", () => {

  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  predictionDiv.innerHTML = `
    <b>Predicted:</b> ? <br/>
    <b>Confidence:</b> ?
  `;
});
