import SwiftUI
import MediaPipeTasksVision

struct HandOverlayView: View {
    @ObservedObject var config: DefaultCostants
    var hands: [[NormalizedLandmark]]
    var originalImageSize: CGSize
    var isPreviewMirrored: Bool = true
    var imageContentMode: UIView.ContentMode = .scaleAspectFill
    var orientation: UIImage.Orientation = .up

    var body: some View {
        GeometryReader { geo in
            Canvas { context, size in
                let (xOffset, yOffset, scaleFactor) = offsetsAndScaleFactor(
                    forImageOfSize: originalImageSize,
                    tobeDrawnInViewOfSize: size,
                    withContentMode: imageContentMode
                )

                for hand in hands {
                    let dots = transformedPoints(
                        for: hand,
                        originalImageSize: originalImageSize,
                        viewSize: size,
                        xOffset: xOffset,
                        yOffset: yOffset,
                        scaleFactor: scaleFactor,
                        orientation: orientation,
                        mirrored: isPreviewMirrored
                    )

                    // Linee (connessioni)
                    var path = Path()
                    for (s, e) in HandConnections.connections {
                        guard s < dots.count, e < dots.count else { continue }
                        path.move(to: dots[s])
                        path.addLine(to: dots[e])
                    }
                    context.stroke(path, with: .color(Color(config.lineColor)), lineWidth: config.lineWidth)

                    // Punti
                    for p in dots {
                        let rect = CGRect(x: p.x - config.pointRadius,
                                          y: p.y - config.pointRadius,
                                          width: config.pointRadius * 2,
                                          height: config.pointRadius * 2)
                        let circle = Path(ellipseIn: rect)
                        context.fill(circle, with: .color(Color(config.pointFillColor)))
                        context.stroke(circle, with: .color(Color(config.pointColor)), lineWidth: 1)
                    }
                }
            }
            .allowsHitTesting(false)
        }
    }

    // MARK: - Trasformazioni coordinate
    private func transformedPoints(
        for landmarks: [NormalizedLandmark],
        originalImageSize: CGSize,
        viewSize: CGSize,
        xOffset: CGFloat,
        yOffset: CGFloat,
        scaleFactor: Double,
        orientation: UIImage.Orientation,
        mirrored: Bool
    ) -> [CGPoint] {
        let oriented: [CGPoint] = landmarks.map { lm in
            let x = CGFloat(lm.x)
            let y = CGFloat(lm.y)

            switch orientation {
            case .up: return CGPoint(x: 1.0 - y, y: x)
            case .down: return CGPoint(x: y, y: 1.0 - x)
            case .left: return CGPoint(x: 1.0 - x, y: 1.0 - y)
            case .right: return CGPoint(x: x, y: y)
            case .upMirrored: return CGPoint(x: y, y: x)
            case .downMirrored: return CGPoint(x: 1.0 - y, y: 1.0 - x)
            case .leftMirrored: return CGPoint(x: x, y: y)
            case .rightMirrored: return CGPoint(x: 1.0 - x, y: y)
            @unknown default: return CGPoint(x: x, y: y)
            }
        }

        var scaled: [CGPoint] = oriented.map { p in
            let px = p.x * viewSize.width
            let py = p.y * viewSize.height

            return CGPoint(x: px, y: py)
        }

        
        if mirrored{
            scaled = scaled.map { pt in
                CGPoint(x: viewSize.width - pt.x, y: pt.y)
            }
        }

        return scaled
    }

    // MARK: - Calcolo scala e offset
    fileprivate func offsetsAndScaleFactor(
        forImageOfSize imageSize: CGSize,
        tobeDrawnInViewOfSize viewSize: CGSize,
        withContentMode contentMode: UIView.ContentMode
    ) -> (xOffset: CGFloat, yOffset: CGFloat, scaleFactor: Double) {
        let widthScale = viewSize.width / imageSize.width
        let heightScale = viewSize.height / imageSize.height

        let scaleFactor: Double = switch contentMode {
        case .scaleAspectFill: max(widthScale, heightScale)
        case .scaleAspectFit: min(widthScale, heightScale)
        default: 1.0
        }

        let scaledSize = CGSize(width: imageSize.width * scaleFactor,
                                height: imageSize.height * scaleFactor)
        let xOffset = (viewSize.width - scaledSize.width) / 2
        let yOffset = (viewSize.height - scaledSize.height) / 2

        return (xOffset, yOffset, scaleFactor)
    }
}
