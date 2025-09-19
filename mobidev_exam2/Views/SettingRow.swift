import SwiftUI

struct SettingRow<Control: View>: View {
    let title: String
    let value: String
    let control: Control
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            HStack {
                Text(value)
                    .foregroundStyle(Color.blue)
                    .fontWeight(.semibold)
                    .frame(width: 50, alignment: .leading)
                
                Spacer()
                
                control
                    .frame(width: 150)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(.systemGray6)))
    }
}
