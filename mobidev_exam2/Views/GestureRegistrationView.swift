import SwiftUI
import Combine

// MARK: - Keyboard Responder
final class KeyboardResponder: ObservableObject {
    @Published var currentHeight: CGFloat = 0
    private var cancellables: Set<AnyCancellable> = []

    init() {
        let willShow = NotificationCenter.default.publisher(for: UIResponder.keyboardWillShowNotification)
            .compactMap { $0.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect }
            .map { $0.height }

        let willHide = NotificationCenter.default.publisher(for: UIResponder.keyboardWillHideNotification)
            .map { _ in CGFloat(0) }

        Publishers.Merge(willShow, willHide)
            .receive(on: DispatchQueue.main)
            .assign(to: \.currentHeight, on: self)
            .store(in: &cancellables)
    }
}

// MARK: - Gesture Registration View
struct GestureRegistrationView: View {
    @ObservedObject var handLandmarkerVM: HandLandmarkerViewModel
    @StateObject private var keyboard = KeyboardResponder()
    
    @FocusState private var isTextFieldFocused: Bool
    @State private var isExpanded: Bool = false
    @State private var isShowingExportModal = false
    
    var currentGestureFrameCount: Int {
        let currentLabel = handLandmarkerVM.gestureLabelToRegister
        return handLandmarkerVM.recordedSamples.filter { $0.label == currentLabel }.count
    }
    
    var body: some View {
        VStack(spacing: 16) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.3)) {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Text("Record Gesture")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .foregroundColor(.white)
                }
                .padding()
                .background(Color.red.opacity(0.7))
                .cornerRadius(10)
            }
            
            if isExpanded {
                VStack(spacing: 12) {
                    // Gesture name input
                    TextField("Gesture name to record", text: $handLandmarkerVM.gestureLabelToRegister)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .focused($isTextFieldFocused)
                        .padding(.horizontal, 4)
                    
                    // Metrics row
                    HStack {
                        MetricItem(
                            title: "Avg Landmarks Presence:",
                            value: "\(handLandmarkerVM.avgPresence * 100)%"
                        )
                        
                        Spacer()
                        
                        MetricItem(
                            title: "Recording Time:",
                            value: "\(String(format: "%.1f s", handLandmarkerVM.totalRecordingTime))",
                            valueColor: handLandmarkerVM.isRecordingGesture ? .red : .primary
                        )
                    }
                    
                    // Frame count and status
                    HStack {
                        MetricItem(
                            title: "Frames Recorded:",
                            value: "\(currentGestureFrameCount)",
                            valueColor: handLandmarkerVM.isRecordingGesture ? .orange : .primary
                        )
                        
                        Spacer()
                        
                        RecordingStatusIndicator(isRecording: handLandmarkerVM.isRecordingGesture)
                    }
                    
                    // Action buttons
                    VStack(spacing: 12) {
                        Button(action: {
                            if handLandmarkerVM.isRecordingGesture {
                                handLandmarkerVM.stopRecordingGesture()
                            } else {
                                handLandmarkerVM.startRecordingGesture(label: handLandmarkerVM.gestureLabelToRegister)
                            }
                        }) {
                            HStack {
                                Image(systemName: handLandmarkerVM.isRecordingGesture ? "stop.circle" : "record.circle")
                                Text(handLandmarkerVM.isRecordingGesture ? "Stop Recording" : "Start Recording")
                            }
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(handLandmarkerVM.isRecordingGesture ? Color.red : Color.green)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                        .disabled(isGestureNameEmpty)
                        .opacity(isGestureNameEmpty ? 0.6 : 1.0)
                        
                        Button(action: {
                            isShowingExportModal = true
                        }) {
                            HStack {
                                Image(systemName: "list.bullet")
                                Text("View Recorded Gestures")
                            }
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                        }
                        .sheet(isPresented: $isShowingExportModal) {
                            ExportGesturesModal(handLandmarkerVM: handLandmarkerVM, isPresented: $isShowingExportModal)
                        }
                    }
                }
                .padding()
                .background(Color.black.opacity(0.3))
                .cornerRadius(10)
                .padding(.bottom, min(keyboard.currentHeight, 150))
                .animation(.easeOut(duration: 0.25), value: keyboard.currentHeight)
            }
        }
        //.padding(.horizontal)
        .background(
            Color.clear
                .contentShape(Rectangle())
                .onTapGesture {
                    hideKeyboard()
                }
        )
    }
    
    private var isGestureNameEmpty: Bool {
        handLandmarkerVM.gestureLabelToRegister.trimmingCharacters(in: .whitespaces).isEmpty
    }
    
    private func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}

struct RecordingStatusIndicator: View {
    let isRecording: Bool
    
    var body: some View {
        HStack(spacing: 6) {
            Circle()
                .fill(isRecording ? Color.red : Color.gray)
                .frame(width: 10, height: 10)
                .scaleEffect(isRecording ? 1.2 : 1.0)
                .animation(
                    isRecording ?
                    Animation.easeInOut(duration: 0.8).repeatForever(autoreverses: true) :
                    .default,
                    value: isRecording
                )
            
            Text(isRecording ? "Recording" : "Idle")
                .font(.caption)
                .foregroundColor(isRecording ? .red : .secondary)
        }
        .padding(8)
        .background(Capsule().fill(Color(.systemGray5)))
    }
}
