import SwiftUI
import AVFoundation

struct CameraView: View {
    @StateObject private var cameraManager = CameraManager()
    @StateObject private var handLandmarkerVM = HandLandmarkerViewModel()
    @StateObject private var gesturePredictor = GesturePredictor()
    
    func runDebugTests() {
        handLandmarkerVM.testPredictionConsistency()
        
        // Verifica gli indici delle feature selezionate
        if let indices = LandmarkUtils.selectedFeatureIndices {
            print("[DEBUGGING] Indici feature selezionate: \(indices)")
            print("[DEBUGGING] Numero di indici: \(indices.count)")
        } else {
            print("[DEBUGGING] Nessun indice di feature selezionate disponibile")
        }
        
        // Verifica le thresholds
        print("[DEBUGGING] Thresholds: \(handLandmarkerVM.confidenceThresholds)")
    }
    var body: some View {
        ZStack {
            if let session = cameraManager.session {
                CameraPreviewView(session: session)
                    .ignoresSafeArea()
            } else {
                Text("Nessuna fotocamera disponibile")
            }

            // Overlay delle mani
            if let imageSize = handLandmarkerVM.currentImageSize {
                ZStack{
                    HandOverlayView(
                        config: handLandmarkerVM.config,
                        hands: handLandmarkerVM.detectedHands,
                        originalImageSize: imageSize,
                        isPreviewMirrored: cameraManager.cameraPosition == .front
                    )
                    .ignoresSafeArea()
                    
                    GeometryReader { geo in
                        ZStack {
                            if let hand = handLandmarkerVM.detectedHands.first {
                                if let box = handLandmarkerVM.boundingBox(
                                    for: hand,
                                    originalSize: handLandmarkerVM.currentImageSize!,
                                    viewSize: geo.size,
                                    isBackCamera: cameraManager.cameraPosition == .back
                                ) {
                                    Rectangle()
                                        .stroke(handLandmarkerVM.isGestureRecognized &&
                                                handLandmarkerVM.gestureRecognized != "Unknown"
                                                ? Color.green : Color.red, lineWidth: 3)
                                        .frame(width: box.width, height: box.height)
                                        .position(x: box.midX, y: box.midY)
                                    Text(
                                        handLandmarkerVM.gestureRecognized != "Unknown" ?
                                        "\(handLandmarkerVM.gestureRecognized) - \(String(format: "%.2f%",handLandmarkerVM.predictionConfidence*100))%"
                                        :
                                        "\(handLandmarkerVM.gestureRecognized)"
                                    )
                                                .font(.callout)
                                                .fontWeight(.bold)
                                                .foregroundColor(.white)
                                                .padding(.horizontal, 6)
                                                .padding(.vertical, 2)
                                                .background(handLandmarkerVM.isGestureRecognized &&
                                                            handLandmarkerVM.gestureRecognized != "Unknown" ? Color.green.opacity(0.85) : Color.red.opacity(0.85))
                                                .position(x: box.midX - 40, y: box.minY - 10)
                                }
                            }
                        }
                    }
                    .ignoresSafeArea()
                }
            }

            VStack{
                Spacer()
                VStack(spacing: 10) {
                       GestureRegistrationView(handLandmarkerVM: handLandmarkerVM)
                       PredictionSheetView(
                            gesturePredictor: gesturePredictor,
                            handLandmarkerVM: handLandmarkerVM
                       )
                       
                       HandBottomSheetView(
                           config: handLandmarkerVM.config,
                           handLandmarkerVM: handLandmarkerVM,
                           switchCameraAction: { cameraManager.switchCamera() }
                       )
                        MetricsBottomSheetView()
                    //Button(action:{runDebugTests()}){Text("button")}
                   }
                   .padding(30)
            }
        }
        .onAppear { cameraManager.start() }
        .onDisappear { cameraManager.stop() }
        .onReceive(cameraManager.$lastSampleBuffer) { sampleBuffer in
            guard let sampleBuffer = sampleBuffer else { return }

            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let height = CVPixelBufferGetHeight(pixelBuffer)
            handLandmarkerVM.currentImageSize = CGSize(width: width, height: height)

            handLandmarkerVM.processFrame(sampleBuffer, orientation: .up)
        }
    }
}
