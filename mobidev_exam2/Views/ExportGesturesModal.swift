import SwiftUI

extension View {
    /// Restituisce il UIViewController corrente dalla gerarchia SwiftUI
    func getRootViewController() -> UIViewController? {
        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let root = scene.windows.first?.rootViewController else {
            return nil
        }
        return root
    }
}

struct ExportGesturesModal: View {
    @ObservedObject var handLandmarkerVM: HandLandmarkerViewModel
    @Binding var isPresented: Bool
    
    @State private var gestureCounts: [(label: String, count: Int)] = []
    @State private var isSharePresented = false
    @State private var shareURL: URL?
    @State private var isLoading = false
    @State private var progressText: String = ""

    var body: some View {
        NavigationView {
            VStack(spacing: 16) {
                List {
                    ForEach(gestureCounts, id: \.label) { item in
                        HStack {
                            Text("\(item.label) (\(item.count) frame)")
                            Spacer()
                            Button(role: .destructive) {
                                handLandmarkerVM.removeSamples(for: item.label)
                                gestureCounts.removeAll { $0.label == item.label }
                            } label: {
                                Image(systemName: "trash")
                            }
                        }
                    }
                }
                
                if isLoading {
                    ProgressView(progressText)
                        .padding()
                }
                
                Button(action: exportDataset) {
                    Text("Esporta Dataset (ZIP)")
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.orange)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .presentShareSheet(items: shareURL != nil ? [shareURL!] : [], isPresented: $isSharePresented)
                .disabled(handLandmarkerVM.recordedSamples.isEmpty || isLoading)
                .opacity(handLandmarkerVM.recordedSamples.isEmpty ? 0.5 : 1)
            }
            .navigationTitle("Gestures registrati")
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Chiudi") {
                        isPresented = false
                    }
                }
            }
            .onAppear {
                gestureCounts = Array(
                    Dictionary(grouping: handLandmarkerVM.recordedSamples, by: { $0.label })
                        .mapValues { $0.count }
                ).map { (label: $0.key, count: $0.value) }
            }
        }
    }
    
    private func exportDataset() {
        guard !handLandmarkerVM.recordedSamples.isEmpty else {
            print("❌ Nessun sample registrato")
            return
        }
        
        isLoading = true
        
        DispatchQueue.global(qos: .userInitiated).async {
            let zipURL = ExportManager.createJSONZip(
                from: handLandmarkerVM.getSamples()
            ) { phase in
                DispatchQueue.main.async {
                    progressText = phase
                }
            }
            
            DispatchQueue.main.async {
                if let zipURL = zipURL {
                    shareURL = zipURL
                    isSharePresented = true
                }
                isLoading = false
            }
        }
    }
}
