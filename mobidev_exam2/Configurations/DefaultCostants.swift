import SwiftUI
import MediaPipeTasksVision

@MainActor
class DefaultCostants: ObservableObject {
    @Published var minFrameDistance: Double = 0.02 

    @Published var numHands: Int = 1
    //Soglia minima di confidenza perché MediaPipe consideri valida la rilevazione iniziale di una mano.
    //Valori bassi (es. 0.1): MediaPipe rileva più facilmente le mani, anche se sono parzialmente visibili o poco nitide
    //Valori alti (es. 0.9): MediaPipe rileva solo mani molto chiare e sicure, rischiando di ignorare mani sfocate o in movimento.
    @Published var minHandDetectionConfidence: Float = 0.5
    //Soglia minima per considerare che una mano sia effettivamente presente in un frame già rilevato.
    //valori bassi: la mano sarà considerata presente anche se alcune landmarks sono poco visibili o la mano è parzialmente fuori campo.
    //Valori alti: MediaPipe potrebbe smettere di considerare la mano presente se la rilevazione è incerta, causando “scomparsa” temporanea della mano.
    @Published var minHandPresenceConfidence: Float = 0.5
    //Soglia minima per il tracking continuo delle mani dopo che sono state rilevate inizialmente.
    //Valori bassi: MediaPipe continuerà a tracciare la mano anche se la confidenza è bassa, aumentando la probabilità di errori nei landmarks
    //Valori alti: il tracking si interrompe se la confidenza scende sotto la soglia, riducendo gli errori ma potenzialmente facendo “scomparire” la mano nei frame difficili.
    @Published var minTrackingConfidence: Float = 0.5

    let modelPath: String? = Bundle.main.path(forResource: "hand_landmarker", ofType: "task")
    @Published var delegate: HandLandmarkerDelegate = .CPU

    let lineWidth: CGFloat = 2
    let pointRadius: CGFloat = 4
    let pointColor: UIColor = .yellow //yellow
    let pointFillColor: UIColor = .red //red
    let lineColor: UIColor = UIColor(red: 0, green: 120/255.0, blue: 120/255.0, alpha: 1)
    //let lineColor: UIColor = UIColor(red: 0, green: 127/255.0, blue: 139/255.0, alpha: 1)
}

enum HandLandmarkerDelegate: CaseIterable {
    case GPU
    case CPU

    var name: String {
        switch self {
        case .GPU: return "GPU"
        case .CPU: return "CPU"
        }
    }

    var delegate: Delegate {
        switch self {
        case .GPU: return .GPU
        case .CPU: return .CPU
        }
    }

    init?(name: String) {
        switch name {
        case HandLandmarkerDelegate.CPU.name: self = .CPU
        case HandLandmarkerDelegate.GPU.name: self = .GPU
        default: return nil
        }
    }
}
