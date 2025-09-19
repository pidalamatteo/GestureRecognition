//
//  CameraFeedService.swift
//  mobidev_exam2
//
//  Created by Matteo on 09/09/25.
//

//prende i frame della camera in tempo reale
import AVFoundation
import UIKit

protocol CameraFeedServiceDelegate: AnyObject {
    func didOutput(sampleBuffer: CMSampleBuffer, orientation: UIImage.Orientation)
    func sessionWasInterrupted()
    func sessionInterruptionEnded()
    func didEncounterSessionRuntimeError()
}

class CameraFeedService: NSObject {
    var cameraPosition: AVCaptureDevice.Position = .front
    private let captureSession = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "CameraFeedSessionQueue")
    
    var session: AVCaptureSession {
        return captureSession
    }
    weak var delegate: CameraFeedServiceDelegate?

    private var videoOutput: AVCaptureVideoDataOutput?

    func startSession() {
        sessionQueue.async {
            self.configureSession(to: self.cameraPosition)
            self.captureSession.startRunning()
        }
    }

    func stopSession() {
        sessionQueue.async {
            if self.captureSession.isRunning {
                self.captureSession.stopRunning()
            }
        }
    }
    
    

    private func configureSession(to cameraPosition: AVCaptureDevice.Position) {
        guard captureSession.inputs.isEmpty else { return }

        captureSession.beginConfiguration()
        captureSession.sessionPreset = .high

        // Input: fotocamera posteriore
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video,
                                                   position: cameraPosition),
            let input = try? AVCaptureDeviceInput(device: device),
            captureSession.canAddInput(input) else {
            print("Errore: impossibile aggiungere input camera")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addInput(input)

        // Output
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "VideoOutputQueue"))
        guard captureSession.canAddOutput(output) else {
            print("Errore: impossibile aggiungere output")
            captureSession.commitConfiguration()
            return
        }
        captureSession.addOutput(output)
        videoOutput = output

        captureSession.commitConfiguration()
    }
   
    
    func switchCamera() {
        sessionQueue.async {
            guard let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput else { return }

            self.captureSession.beginConfiguration()
            self.captureSession.removeInput(currentInput)

            // cambia posizione
            self.cameraPosition = (self.cameraPosition == .front) ? .back : .front

            // riusa configureSession passando cameraPosition
            self.configureSession(to: self.cameraPosition)

            self.captureSession.commitConfiguration()
        }
    }
     
    /*
    func imageOrientation(from deviceOrientation: UIDeviceOrientation) -> UIImage.Orientation {
        switch deviceOrientation {
        case .portrait: return .right
        case .portraitUpsideDown: return .left
        case .landscapeLeft: return .up
        case .landscapeRight: return .down
        default: return .up
        }
    }
    func imageOrientation(fromRotationAngle angle: Int32) -> UIImage.Orientation {
        switch angle {
        case 0: return .right           // Portrait
        case 90: return .down           // Landscape Left
        case 180: return .left          // Portrait Upside Down
        case 270: return .up            // Landscape Right
        default: return .up
        }
    }
     */
}

extension CameraFeedService: AVCaptureVideoDataOutputSampleBufferDelegate {
    func currentOrientation(for position: AVCaptureDevice.Position) -> UIImage.Orientation{
        switch UIDevice.current.orientation {
        case .portrait: return position == .front ? .leftMirrored: .right
        case .portraitUpsideDown: return position == .front ? .rightMirrored: .left
        case .landscapeLeft: return position == .front ? .downMirrored: .up
        case .landscapeRight: return position == .front ? .upMirrored: .down
        default: return .up
        }
    }
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
    
        let orientation = currentOrientation(for: cameraPosition)
        delegate?.didOutput(sampleBuffer: sampleBuffer, orientation: orientation)
    }

}

