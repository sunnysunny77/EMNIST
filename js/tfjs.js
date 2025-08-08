import * as tf from "@tensorflow/tfjs";

const emnistLabels = [
  "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
];

let model;
let drawing = false;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const CANVAS_SIZE = 280;
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
    const [maxIndex, maxVal] = tf.tidy(() => {
      const img = tf.browser.fromPixels(canvas, 1);
      const inverted = tf.sub(255, img).div(255.0);
      const resized = tf.image.resizeBilinear(inverted, [28, 28]);
      const batched = resized.expandDims(0);
      const prediction = model.predict(batched);
      const values = prediction.dataSync();
      const maxVal = Math.max(...values);
      const maxIndex = values.indexOf(maxVal);

      return [maxIndex, maxVal];
    });

    predictionDiv.innerText = `Prediction: ${emnistLabels[maxIndex]} (Confidence: ${maxVal.toFixed(4)})`;
  } catch (error) {
    predictionDiv.textContent = `Error during prediction: ${error.message}`;
    console.error(error);
  }
};

const getCanvasCoords = (e) => {

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
