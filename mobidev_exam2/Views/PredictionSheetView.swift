import SwiftUI

struct PredictionSheetView: View {
    @ObservedObject var gesturePredictor: GesturePredictor
    @ObservedObject var handLandmarkerVM: HandLandmarkerViewModel
    @State private var isExpanded: Bool = false
    
    // Binding sicuri per le proprietà config
    private var minStableFrames: Binding<Int> {
        Binding(
            get: { gesturePredictor.config.minStableFrames },
            set: { newValue in
                DispatchQueue.main.async {
                    gesturePredictor.config.minStableFrames = newValue
                    gesturePredictor.resetTemporalHistory() // Reset quando cambia
                }
            }
        )
    }
    
    private var requiredConsensusRatio: Binding<Double> {
        Binding(
            get: { gesturePredictor.config.requiredConsensusRatio },
            set: { newValue in
                DispatchQueue.main.async {
                    gesturePredictor.config.requiredConsensusRatio = newValue
                    gesturePredictor.resetTemporalHistory() // Reset quando cambia
                }
            }
        )
    }
    
    private var minConfidenceThreshold: Binding<Double> {
        Binding(
            get: { gesturePredictor.config.minConfidenceThreshold },
            set: { newValue in
                DispatchQueue.main.async {
                    gesturePredictor.config.minConfidenceThreshold = newValue
                    gesturePredictor.resetTemporalHistory() // Reset quando cambia
                }
            }
        )
    }
    
    private var timeWindow: Binding<Double> {
        Binding(
            get: { gesturePredictor.config.timeWindow },
            set: { newValue in
                DispatchQueue.main.async {
                    gesturePredictor.config.timeWindow = newValue
                    gesturePredictor.resetTemporalHistory() // Reset quando cambia
                }
            }
        )
    }

    var body: some View {
        VStack(spacing: 16) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.3)) {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Text("Settings")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .foregroundColor(.white)
                }
                .padding()
                .background(Color.blue.opacity(0.7))
                .cornerRadius(10)
            }

            if isExpanded {
                VStack(spacing: 12) {
                    // Performance metrics
                    HStack{
                        VStack{
                            Text("Frame Interval:")
                                .font(.caption)
                            Text(String(format: "%.2f ms", handLandmarkerVM.frameInterval))
                                .foregroundStyle(Color.orange)
                                .fontWeight(.semibold)
                        }
                        .padding(10)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
                        
                        Spacer()
                        
                        VStack{
                            Text("Inference Time:")
                                .font(.caption)
                            Text(String(format: "%.2f ms", handLandmarkerVM.inferenceTime))
                                .foregroundStyle(Color.orange)
                                .fontWeight(.semibold)
                        }
                        .padding(10)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
                    }
                    
                    // Min Stable Frames
                    SettingRow(
                        title: "Min Stable Frames:",
                        value: "\(gesturePredictor.config.minStableFrames)",
                        control: Stepper("", value: minStableFrames, in: 1...10)
                            .labelsHidden()
                    )
                    
                    // Required Consensus Ratio
                    SettingRow(
                        title: "Required Consensus Ratio:",
                        value: String(format: "%.2f", gesturePredictor.config.requiredConsensusRatio),
                        control: Slider(
                            value: requiredConsensusRatio,
                            in: 0.05...0.95,
                            step: 0.05
                        )
                    )
                    
                    // Min Confidence Threshold
                    SettingRow(
                        title: "Min Confidence Threshold:",
                        value: String(format: "%.2f", gesturePredictor.config.minConfidenceThreshold),
                        control: Slider(
                            value: minConfidenceThreshold,
                            in: 0.05...1.0,
                            step: 0.05
                        )
                    )
                    
                    // Time Window
                    SettingRow(
                        title: "Time Window:",
                        value: String(format: "%.3f", gesturePredictor.config.timeWindow),
                        control: Slider(
                            value: timeWindow,
                            in: 0.01...0.1,
                            step: 0.005
                        )
                    )
                    
                    // Reset button
                    Button(action: {
                        resetToDefaults()
                    }) {
                        Text("Reset to Defaults")
                            .foregroundColor(.white)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 8)
                            .background(Color.red.opacity(0.8))
                            .cornerRadius(8)
                    }
                    .padding(.top, 8)
                }
                .padding()
                .background(Color.black.opacity(0.3))
                .cornerRadius(10)
            }
        }
        //.padding(.horizontal)
    }
    
    private func resetToDefaults() {
        let defaults = GesturePredictor.SmoothingConfig()
        
        DispatchQueue.main.async {
            gesturePredictor.config.minStableFrames = defaults.minStableFrames
            gesturePredictor.config.requiredConsensusRatio = defaults.requiredConsensusRatio
            gesturePredictor.config.minConfidenceThreshold = defaults.minConfidenceThreshold
            gesturePredictor.config.timeWindow = defaults.timeWindow
            gesturePredictor.resetTemporalHistory()
        }
    }
}
