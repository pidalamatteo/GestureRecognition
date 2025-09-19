import SwiftUI

/*
struct CardMetrics: View {
    let metrics: MetricsStruct.MetricsData
    
    var body: some View {
        VStack(spacing: 16) {
            // Metriche generali
            ScrollView{
                HStack(spacing: 20) {
                    VStack {
                        Text("Accuracy")
                            .font(.headline)
                        Text(String(format: "%.2f%%", metrics.accuracy * 100))
                            .font(.title2)
                            .foregroundColor(.blue)
                    }
                    
                    VStack {
                        Text("Balanced Acc")
                            .font(.headline)
                        Text(String(format: "%.2f%%", metrics.balancedAccuracy * 100))
                            .font(.title2)
                            .foregroundColor(.green)
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 12).fill(Color(.systemGray6)))
                
                // Metriche per classe
                Text("Metriche per Classe")
                    .font(.headline)
                
                ForEach(metrics.classMetrics.keys.sorted(), id: \.self) { className in
                    if let classMetric = metrics.classMetrics[className] {
                        VStack(alignment: .leading) {
                            Text(className)
                                .font(.headline)
                                .padding(.bottom, 4)
                            
                            HStack {
                                VStack {
                                    Text("Precision")
                                        .font(.caption)
                                    Text(String(format: "%.3f", classMetric.precision))
                                        .font(.subheadline)
                                }
                                
                                Spacer()
                                
                                VStack {
                                    Text("Recall")
                                        .font(.caption)
                                    Text(String(format: "%.3f", classMetric.recall))
                                        .font(.subheadline)
                                }
                                
                                Spacer()
                                
                                VStack {
                                    Text("F1-Score")
                                        .font(.caption)
                                    Text(String(format: "%.3f", classMetric.f1))
                                        .font(.subheadline)
                                }
                            }
                        }
                        .padding()
                        .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
                    }
                }
            }
            .scrollIndicators(.hidden)
        }
        .padding()
    }
}
*/
