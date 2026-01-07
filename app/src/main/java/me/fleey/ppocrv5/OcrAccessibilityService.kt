/*
 * Copyright (C) 2025 Fleey
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package me.fleey.ppocrv5

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.AccessibilityServiceInfo
import android.content.*
import android.graphics.Bitmap
import android.os.Build
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import me.fleey.ppocrv5.ocr.AcceleratorType
import me.fleey.ppocrv5.ocr.OcrEngine
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.WindowManager
import android.util.DisplayMetrics
import android.graphics.PixelFormat
import android.hardware.HardwareBuffer
import java.util.concurrent.TimeUnit
import android.os.Handler
import android.os.Looper

class OcrAccessibilityService : AccessibilityService() {

    private var ocrEngine: OcrEngine? = null
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val handler = Handler(Looper.getMainLooper())

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action == "me.fleey.ppocrv5.ACTION_OCR_SCREEN") {
                takeScreenshotAndProcess()
            }
        }
    }

    override fun onCreate() {
        super.onCreate()
        executor.execute {
            ocrEngine = OcrEngine.create(this, AcceleratorType.GPU).getOrNull()
        }
        val filter = IntentFilter("me.fleey.ppocrv5.ACTION_OCR_SCREEN")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            registerReceiver(receiver, filter)
        }
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {}

    override fun onInterrupt() {}

    private fun takeScreenshotAndProcess() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            takeScreenshot(Display.DEFAULT_DISPLAY, executor, object : TakeScreenshotCallback {
                override fun onSuccess(screenshotResult: ScreenshotResult) {
                    val bitmap = Bitmap.wrapHardwareBuffer(
                        screenshotResult.hardwareBuffer,
                        screenshotResult.colorSpace
                    )?.copy(Bitmap.Config.ARGB_8888, false)
                    screenshotResult.hardwareBuffer.close()
                    if (bitmap != null) {
                        processBitmap(bitmap)
                    }
                }

                override fun onFailure(errorCode: Int) {
                    // Handle failure
                }
            })
        }
    }

    private fun processBitmap(bitmap: Bitmap) {
        val engine = ocrEngine ?: return
        val results = engine.process(bitmap)
        val jsonOutput = results.map {
            OcrJsonResult(
                text = it.text,
                confidence = it.confidence,
                x = it.centerX,
                y = it.centerY,
                w = it.width,
                h = it.height
            )
        }

        val jsonString = Json.encodeToString(jsonOutput)
        val file = File(getExternalFilesDir(null), "ocr_result.json")
        file.writeText(jsonString)

        copyToClipboard(file.absolutePath)
    }

    private fun copyToClipboard(text: String) {
        handler.post {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("OCR Result Path", text)
            clipboard.setPrimaryClip(clip)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        unregisterReceiver(receiver)
        ocrEngine?.close()
        executor.shutdown()
    }

    @Serializable
    data class OcrJsonResult(
        val text: String,
        val confidence: Float,
        val x: Float,
        val y: Float,
        val w: Float,
        val h: Float
    )
}
