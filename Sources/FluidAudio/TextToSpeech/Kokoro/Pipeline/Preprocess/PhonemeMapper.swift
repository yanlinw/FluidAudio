import Foundation
import OSLog

enum PhonemeMapper {
    private static let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "PhonemeMapper")

    /// Map a sequence of IPA tokens to Kokoro vocabulary tokens, filtering to `allowed`.
    /// Unknown symbols are approximated when possible; otherwise dropped.
    static func mapIPA(_ ipaTokens: [String], allowed: Set<String>) -> [String] {
        var out: [String] = []
        var index = ipaTokens.startIndex
        var droppedTokens: [String] = []

        while index < ipaTokens.count {
            let token = ipaTokens[index]

            if allowed.contains(token) {
                out.append(token)
                index += 1
                continue
            }

            // Try pair merge for affricates represented as separate chars
            if index + 1 < ipaTokens.count {
                let pair = token + ipaTokens[index + 1]
                if let mapped = mapSingle(pair, allowed: allowed) {
                    out.append(mapped)
                    index += 2
                    continue
                }
            }

            if let mapped = mapSingle(token, allowed: allowed) {
                out.append(mapped)
            } else {
                droppedTokens.append(token)
            }
            index += 1
        }

        if !droppedTokens.isEmpty {
            logger.debug("Dropped IPA tokens not in vocabulary: \(droppedTokens)")
        }

        return out
    }

    private static func mapSingle(_ raw: String, allowed: Set<String>) -> String? {
        // If stress/length/diacritics are used and in vocab, pass-through
        if allowed.contains(raw) { return raw }

        // Normalize some IPA to approximate Kokoro inventory
        let ipaToKokoro = Self.ipaTable

        if let mapped = ipaToKokoro[raw] {
            // Empty string means "drop silently" (e.g., tone markers)
            if mapped.isEmpty { return "" }
            if allowed.contains(mapped) { return mapped }
        }

        // Simple latin fallback: map ascii letters and digits if they exist
        if raw.count == 1, let scalar = raw.unicodeScalars.first,
            asciiLetterOrDigit.contains(scalar)
        {
            let s = String(raw)
            if allowed.contains(s) { return s }
        }
        return nil
    }

    private static let asciiLetterOrDigit: CharacterSet = {
        var set = CharacterSet(charactersIn: "0"..."9")
        set.formUnion(CharacterSet(charactersIn: "a"..."z"))
        set.formUnion(CharacterSet(charactersIn: "A"..."Z"))
        return set
    }()

    private static let ipaTable: [String: String] = [
        // Affricates
        "t͡ʃ": "ʧ", "tʃ": "ʧ", "d͡ʒ": "ʤ", "dʒ": "ʤ",
        // Fricatives
        "ʃ": "ʃ", "ʒ": "ʒ", "θ": "θ", "ð": "ð",
        // Approximants / alveolars
        "ɹ": "r", "ɾ": "t", "ɫ": "l",
        // Nasals
        "ŋ": "ŋ",
        // Vowels
        "æ": "æ", "ɑ": "ɑ", "ɒ": "ɑ", "ʌ": "ʌ",
        "ɪ": "ɪ", "i": "i", "ʊ": "ʊ", "u": "u",
        "ə": "ə", "ɚ": "ɚ", "ɝ": "ɝ",
        "ɛ": "ɛ", "e": "e", "o": "o", "ɔ": "ɔ",
        // Diphthongs
        "eɪ": "e", "oʊ": "o", "aɪ": "a", "aʊ": "a", "ɔɪ": "ɔ",
        // Mandarin tone markers (eSpeak numeric format) - drop silently for now
        // Kokoro model may encode tones implicitly via voice embeddings
        "1": "", "2": "", "3": "", "4": "", "5": "",
        // Mandarin-specific IPA symbols
        "ɕ": "ɕ", "ʈʂ": "ʈʂ", "ʐ": "ʐ", "χ": "χ", "ɻ": "ɻ",
        "y": "y", "ɥ": "ɥ", "ɤ": "ɤ", "ɚ": "ɚ", "ɜ": "ɜ",
    ]
}
