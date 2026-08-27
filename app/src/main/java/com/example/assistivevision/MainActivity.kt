package com.example.assistivevision

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Paint
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.assistivevision.ui.theme.AssistiveVisionTheme
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import android.speech.tts.TextToSpeech
import java.util.Locale

data class Detection(
    val label: String,
    val confidence: Float,
    val left: Float,   // normalized 0..1
    val top: Float,
    val right: Float,
    val bottom: Float
)

class MainActivity : ComponentActivity() {

    private var hasCameraPermission = false

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
        recreate()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        hasCameraPermission = ContextCompat.checkSelfPermission(
            this, Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (!hasCameraPermission) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            AssistiveVisionTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    if (hasCameraPermission) {
                        CameraPreviewScreen(modifier = Modifier.fillMaxSize())
                    } else {
                        Text(
                            text = "Camera permission is required.",
                            modifier = Modifier.fillMaxSize()
                        )
                    }
                }
            }
        }
    }
}

fun loadModelFile(context: Context, fileName: String): MappedByteBuffer {
    val fileDescriptor = context.assets.openFd(fileName)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
}

fun loadLabels(context: Context): List<String> {
    return context.assets.open("labels.txt").bufferedReader().readLines()
}

// Simple greedy Non-Max Suppression to remove overlapping duplicate boxes
fun nonMaxSuppression(detections: List<Detection>, iouThreshold: Float = 0.45f): List<Detection> {
    val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
    val result = mutableListOf<Detection>()

    while (sorted.isNotEmpty()) {
        val best = sorted.removeAt(0)
        result.add(best)
        sorted.removeAll { other ->
            val iou = calculateIoU(best, other)
            iou > iouThreshold
        }
    }
    return result
}

fun calculateIoU(a: Detection, b: Detection): Float {
    val interLeft = max(a.left, b.left)
    val interTop = max(a.top, b.top)
    val interRight = min(a.right, b.right)
    val interBottom = min(a.bottom, b.bottom)

    val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
    val areaA = (a.right - a.left) * (a.bottom - a.top)
    val areaB = (b.right - b.left) * (b.bottom - b.top)

    return if (areaA + areaB - interArea <= 0f) 0f else interArea / (areaA + areaB - interArea)
}

@Composable
fun CameraPreviewScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current

    var detections by remember { mutableStateOf(listOf<Detection>()) }
    val labels = remember { loadLabels(context) }
    val lastSpokenTime = remember { mutableStateOf(0L) }
    val lastSpokenLabel = remember { mutableStateOf("") }

    val tts = remember {
        var ttsInstance: TextToSpeech? = null
        ttsInstance = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ttsInstance?.language = Locale.US
            }
        }
        ttsInstance
    }

    Box(modifier = modifier) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { ctx ->
                val previewView = PreviewView(ctx)
                val interpreter = Interpreter(loadModelFile(ctx, "best.tflite"))
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    var frameCount = 0

                    imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(ctx)) { imageProxy: ImageProxy ->
                        frameCount++
                        if (frameCount % 5 == 0) { // run every 5th frame for smoother feel
                            try {
                                val bitmap = imageProxy.toBitmap()
                                val resized = Bitmap.createScaledBitmap(bitmap, 320, 320, true)

                                // Channel-first input: [1, 3, 320, 320]
                                val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * 320 * 320 * 4)
                                inputBuffer.order(ByteOrder.nativeOrder())

                                val pixels = IntArray(320 * 320)
                                resized.getPixels(pixels, 0, 320, 0, 0, 320, 320)

                                // R channel plane
                                for (p in pixels) inputBuffer.putFloat(((p shr 16) and 0xFF) / 255.0f)
                                // G channel plane
                                for (p in pixels) inputBuffer.putFloat(((p shr 8) and 0xFF) / 255.0f)
                                // B channel plane
                                for (p in pixels) inputBuffer.putFloat((p and 0xFF) / 255.0f)

                                val output = Array(1) { Array(98) { FloatArray(2100) } }
                                interpreter.run(inputBuffer, output)

                                val newDetections = mutableListOf<Detection>()
                                for (j in 0 until 2100) {
                                    val cx = output[0][0][j]
                                    val cy = output[0][1][j]
                                    val w = output[0][2][j]
                                    val h = output[0][3][j]

                                    var bestScore = 0f
                                    var bestClass = -1
                                    for (c in 0 until 94) {
                                        val score = output[0][4 + c][j]
                                        if (score > bestScore) {
                                            bestScore = score
                                            bestClass = c
                                        }
                                    }

                                    if (bestScore > 0.4f && bestClass in labels.indices) {
                                        val left = (cx - w / 2f).coerceIn(0f, 1f)
                                        val top = (cy - h / 2f).coerceIn(0f, 1f)
                                        val right = (cx + w / 2f).coerceIn(0f, 1f)
                                        val bottom = (cy + h / 2f).coerceIn(0f, 1f)
                                        newDetections.add(
                                            Detection(labels[bestClass], bestScore, left, top, right, bottom)
                                        )
                                    }
                                }

                                detections = nonMaxSuppression(newDetections)

                                if (detections.isNotEmpty()) {
                                    val topDetection = detections.maxByOrNull { it.confidence }
                                    if (topDetection != null) {
                                        val now = System.currentTimeMillis()
                                        val cooldownMs = 3000L // don't repeat the same word more than once every 3 seconds
                                        val isNewLabel = topDetection.label != lastSpokenLabel.value
                                        val cooldownPassed = now - lastSpokenTime.value > cooldownMs

                                        if (isNewLabel || cooldownPassed) {
                                            tts?.speak(topDetection.label, TextToSpeech.QUEUE_FLUSH, null, null)
                                            lastSpokenLabel.value = topDetection.label
                                            lastSpokenTime.value = now
                                        }
                                    }
                                    Log.d("ASSISTIVE_VISION", "Detected: ${detections.map { "${it.label} (${it.confidence})" }}")
                                }
                            } catch (e: Exception) {
                                Log.e("ASSISTIVE_VISION", "Inference error: ${e.message}")
                            }
                        }
                        imageProxy.close()
                    }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis
                    )
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            }
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height

            detections.forEach { det ->
                val left = det.left * canvasWidth
                val top = det.top * canvasHeight
                val right = det.right * canvasWidth
                val bottom = det.bottom * canvasHeight

                drawRect(
                    color = Color.Red,
                    topLeft = Offset(left, top),
                    size = Size(right - left, bottom - top),
                    style = Stroke(width = 4f)
                )

                drawContext.canvas.nativeCanvas.drawText(
                    "${det.label} ${(det.confidence * 100).toInt()}%",
                    left,
                    (top - 10).coerceAtLeast(20f),
                    Paint().apply {
                        color = android.graphics.Color.RED
                        textSize = 36f
                        isFakeBoldText = true
                    }
                )
            }
        }
        androidx.compose.runtime.DisposableEffect(Unit) {
            onDispose {
                tts?.stop()
                tts?.shutdown()
            }
        }
    }
}