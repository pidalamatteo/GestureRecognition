import SwiftUI
/*
class MetricsViewModel: ObservableObject {
    @Published var metricsStruct: MetricsStruct?
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var selectedTab = 0
    
    func loadMetrics(from url: URL) {
        isLoading = true
        errorMessage = nil
        
        DispatchQueue.global(qos: .background).async {
            do {
                let data = try Data(contentsOf: url)
                let decoder = JSONDecoder()
                let metrics = try decoder.decode(MetricsStruct.self, from: data)
                
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
*/
