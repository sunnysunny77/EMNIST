<!DOCTYPE html>
<html lang="en" data-overlayscrollbars-initialize>
<head>
  <meta charset="utf-8" />
  <meta name="description" content="ENMIST" />
  <meta name="keywords" content="ENMIST" />
  <meta name="author" content="D>C" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ENMIST</title>
  <link href="./css/app.min.css" rel="stylesheet" type="text/css" />
  <link rel="manifest" href="manifest.json" />
  <link rel="apple-touch-icon" href="images/pwa-logo-small.webp" />
</head>

<body data-overlayscrollbars-initialize>

  <div class="container" id="container">

    <h1>EMNIST Handwritten character recognition</h1>

    <canvas id="canvas"></canvas>

    <div class="buttons">

      <button id="clearBtn">Clear</button>

      <button id="predictBtn">Predict</button>

    </div>

    <div id="prediction">

      Loading model...

    </div>

  </div>

  <script src="./js/app.min.js" defer></script>

</body>

</html>