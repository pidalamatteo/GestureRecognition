import SwiftUI

struct MetricItem: View {
    let title: String
    let value: String
    var valueColor: Color = .primary
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(value)
                .font(.system(.body, design: .monospaced))
                .foregroundColor(valueColor)
                .fontWeight(.medium)
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
    }
}
