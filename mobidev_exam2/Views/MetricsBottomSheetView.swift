import SwiftUI

struct MetricsBottomSheetView: View {
    @StateObject private var metricsModel = MetricsViewModel()
    @State private var isExpanded: Bool = false
    
    var body: some View {
        VStack(spacing: 16) {
            Button(action: {
                withAnimation(.easeInOut(duration: 0.3)) {
                    isExpanded.toggle()
                    
                    // Carica le metriche quando si espande
                    if isExpanded && metricsModel.metricsStruct == nil {
                        if let url = Bundle.main.url(forResource: "metrics", withExtension: "json") {
                            metricsModel.loadMetrics(from: url)
                        }
                    }
                }
            }) {
                HStack {
                    Text("Model Metrics")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.up")
                        .foregroundColor(.white)
                }
                .padding()
                .background(Color.orange.opacity(0.7))
                .cornerRadius(10)
            }

            if isExpanded {
                VStack(spacing: 12) {
                    if metricsModel.isLoading {
                        ProgressView("Loading metrics...")
                            .padding()
                            .frame(maxWidth: .infinity)
                    } else if let data = metricsModel.metricsStruct {
                        // Picker per scegliere tra validation e testing
                        Picker("Dataset", selection: $metricsModel.selectedTab) {
                            Text("Validation").tag(0)
                            Text("Testing").tag(1)
                        }
                        .pickerStyle(SegmentedPickerStyle())
                        .padding(.horizontal)
                        
                        // Mostra le metriche in base alla selezione
                        if metricsModel.selectedTab == 0 {
                            CardMetrics(metrics: data.validation)
                        } else {
                            CardMetrics(metrics: data.testing)
                        }
                    } else if let error = metricsModel.errorMessage {
                        Text("Error: \(error)")
                            .foregroundColor(.red)
                            .padding()
                    }
                }
                .padding()
                .background(Color.black.opacity(0.3))
                .cornerRadius(10)
            }
        }
        //.padding(.horizontal)
    }
}

// ViewModel per le metriche
class MetricsViewModel: ObservableObject {
    @Published var metricsStruct: MetricsStruct?
    @Published var isLoading: Bool = false
    @Published var errorMessage: String?
    @Published var selectedTab: Int = 0
    
    func loadMetrics(from url: URL) {
        isLoading = true
        errorMessage = nil
        
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let data = try Data(contentsOf: url)
                let metrics = try JSONDecoder().decode(MetricsStruct.self, from: data)
                
                DispatchQueue.main.async {
                    self.metricsStruct = metrics
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
}

// Vista per visualizzare le metriche
struct CardMetrics: View {
    let metrics: MetricsStruct.MetricsData
    
    // Calcola le metriche medie per tutte le classi
    private var averageMetrics: (precision: Double, recall: Double, f1: Double) {
        let classCount = Double(metrics.classMetrics.count)
        guard classCount > 0 else { return (0, 0, 0) }
        
        let totalPrecision = metrics.classMetrics.values.reduce(0) { $0 + $1.precision }
        let totalRecall = metrics.classMetrics.values.reduce(0) { $0 + $1.recall }
        let totalF1 = metrics.classMetrics.values.reduce(0) { $0 + $1.f1 }
        
        return (
            totalPrecision / classCount,
            totalRecall / classCount,
            totalF1 / classCount
        )
    }
    
    var body: some View {
        VStack(spacing: 12) {
            // Metriche generali
            MetricRow(title: "Accuracy", value: String(format: "%.2f%%", metrics.accuracy * 100))
            MetricRow(title: "Balanced Accuracy", value: String(format: "%.2f%%", metrics.balancedAccuracy * 100))
            
            // Metriche medie per classe
            MetricRow(title: "Avg Precision", value: String(format: "%.2f%%", averageMetrics.precision * 100))
            MetricRow(title: "Avg Recall", value: String(format: "%.2f%%", averageMetrics.recall * 100))
            MetricRow(title: "Avg F1 Score", value: String(format: "%.2f", averageMetrics.f1))
            
            // Selettore per visualizzare metriche specifiche per classe
            ClassMetricsSelector(classMetrics: metrics.classMetrics, confusionMatrix: metrics.confusionMatrix)
        }
        .padding()
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(.systemGray6)))
    }
}

// Selettore per visualizzare metriche specifiche per classe
struct ClassMetricsSelector: View {
    let classMetrics: [String: MetricsStruct.ClassMetric]
    let confusionMatrix: [String: [String: Int]]
    @State private var selectedClass: String = ""
    
    var body: some View {
        VStack(spacing: 12) {
            ScrollView{
                if !classMetrics.isEmpty {
                    Text("Class-specific Metrics")
                        .font(.headline)
                        .padding(.top, 8)
                    
                    Picker("Select Class", selection: $selectedClass) {
                        ForEach(Array(classMetrics.keys.sorted()), id: \.self) { className in
                            Text(className).tag(className)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onAppear {
                        if selectedClass.isEmpty {
                            selectedClass = Array(classMetrics.keys.sorted()).first ?? ""
                        }
                    }
                    
                    if let metrics = classMetrics[selectedClass] {
                        VStack(spacing: 8) {
                            MetricRow(title: "Precision", value: String(format: "%.2f%%", metrics.precision * 100))
                            MetricRow(title: "Recall", value: String(format: "%.2f%%", metrics.recall * 100))
                            MetricRow(title: "F1 Score", value: String(format: "%.2f", metrics.f1))
                        }
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray5)))
                    }
                    
                    // Matrice di confusione per la classe selezionata
                    if let row = confusionMatrix[selectedClass] {
                        Text("Confusion for \(selectedClass)")
                            .font(.subheadline)
                            .padding(.top, 8)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: min(row.count, 4)), spacing: 4) {
                            ForEach(Array(row.keys.sorted()), id: \.self) { predictedClass in
                                if let count = row[predictedClass] {
                                    VStack {
                                        Text(predictedClass)
                                            .font(.caption2)
                                        Text("\(count)")
                                            .font(.system(size: 14, weight: .bold))
                                    }
                                    .frame(maxWidth: .infinity)
                                    .padding(4)
                                    .background(Color.gray.opacity(0.2))
                                    .cornerRadius(4)
                                }
                            }
                        }
                    }
                }
            }.scrollIndicators(.hidden)
        }
    }
}

// Componente per visualizzare una metrica
struct MetricRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .font(.subheadline)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.system(.body, design: .monospaced))
                .foregroundColor(.primary)
        }
    }
}
