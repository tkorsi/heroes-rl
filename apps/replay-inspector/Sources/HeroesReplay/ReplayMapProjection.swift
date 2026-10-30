// Presentation uses only objects still present in the current native state.
// A stale visibility ID must never resurrect a collected or defeated object.
enum ReplayMapProjection {
    static func visibleObjects(_ objects: [[Int]], ids: [Int]) -> [[Int]] {
        let known = Set(ids)
        return objects.filter { $0.count >= 7 && known.contains($0[0]) }
    }

    static func mineOwner(_ object: [Int]) -> Int? {
        guard object.count >= 7, object[4] == 53, object[6] == 0 || object[6] == 1 else { return nil }
        return object[6]
    }
}
