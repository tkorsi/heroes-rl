import SwiftUI
import AppKit

struct Header: Decodable { let width: Int; let height: Int; let terrain: [Int] }
struct PlayerView: Decodable { let visible: String; let objects: [Int] }
struct Frame: Decodable { let seq: Int; let day: Int; let boundary: String?; let objects: [[Int]]; let players: [String: PlayerView]; let battles: [[Int]] }
struct Slot: Decodable { let slot: Int; let creature: Int; let count: Int }
struct Army: Decodable { let name: String; let status: String; let position: [Int]?; let slots: [Slot]; let source_packet: Int; let freshness: String }
struct Event: Decodable, Identifiable { let id: Int; let title: String; let actor: String; let category: String; let frame: Int; let packet: Int; let source: String; let raw: String; let observation: String; let policy: String; let summary: String; let army: Army; let hero_details: String?; let hero_details_packet: Int? }
struct ObjectDetails: Decodable { let initial: String; let quantity: Int?; let subtype: String }
struct Replay: Decodable {
    let object_details: [String:ObjectDetails]?
    let schema: String; let title: String; let subtitle: String; let header: Header
    let frames: [Frame]; let armies: [Army]; let events: [Event]; let documents: [String: String]
    func validate() throws {
        guard schema == "heroes-replay-inspector-v1", !frames.isEmpty, armies.count == frames.count,
              header.width > 0, header.height > 0, header.width <= 256, header.height <= 256,
              header.terrain.count == header.width * header.height, !events.isEmpty,
              events.enumerated().allSatisfy({ $0.offset == $0.element.id && frames.indices.contains($0.element.frame) }),
              frames.allSatisfy({ frame in frame.objects.allSatisfy { $0.count >= 7 } && frame.battles.allSatisfy { $0.count >= 5 } && ["0", "1"].allSatisfy { frame.players[$0]?.visible.count == header.width * header.height } })
        else { throw NSError(domain: "Replay", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid or unsupported replay bundle"]) }
    }
}
@MainActor final class Store: ObservableObject {
    @Published var replay: Replay?
    @Published var selected = 0
    @Published var mode = "Actions"
    @Published var query = ""
    @Published var error = ""
    @Published var detail = "Event"
    @Published var document = "manifest.json"
    var filtered: [Event] {
        replay?.events.filter { e in
            (mode == "All events" || (mode == "Decisions" ? e.category == "decision" : e.category == "state" || e.category == "decision")) &&
            (query.isEmpty || "\(e.title) \(e.actor) \(e.source)".localizedCaseInsensitiveContains(query))
        } ?? []
    }
    var event: Event? { replay?.events[safe: selected] }
    func move(_ delta: Int) {
        let list = filtered
        if delta > 0, let next = list.first(where: { $0.id > selected }) { selected = next.id }
        if delta < 0, let next = list.last(where: { $0.id < selected }) { selected = next.id }
    }
    func load(_ url: URL) {
        do {
            let value = try JSONDecoder().decode(Replay.self, from: Data(contentsOf: url))
            try value.validate()
            replay = value; selected = 0; error = ""; query = ""; mode = "Actions"
        } catch { self.error = error.localizedDescription }
    }
    func open() {
        let panel = NSOpenPanel(); panel.allowedContentTypes = [.json]; panel.canChooseDirectories = false
        if panel.runModal() == .OK, let url = panel.url { load(url) }
    }
}
extension Collection { subscript(safe index: Index) -> Element? { indices.contains(index) ? self[index] : nil } }
let navy = Color(red: 0.063, green: 0.09, blue: 0.133)
let card = Color(red: 0.094, green: 0.137, blue: 0.196)
let blue = Color(red: 0.39, green: 0.65, blue: 1)
let red = Color(red: 1, green: 0.47, blue: 0.49)

// Semantic map symbols remain local and require no original game assets.
func objectAppearance(_ object: [Int]) -> (icon: String, name: String) {
    switch object[4] {
    case 34: return ("♞", object[5] == 2 ? "Герой Edric" : object[5] == 0 ? "Герой Orrin" : "Герой")
    case 98: return ("🏰", "Город")
    case 53: return ("⛏️", "Шахта")
    case 54: return ("🛡️", object[5] == 0 ? "Нейтральные копейщики" : "Нейтральная армия")
    case 79:
        let icons = ["🪵", "🧪", "🪨", "🟡", "🔮", "💎", "💰"]
        let names = ["Дерево", "Ртуть", "Руда", "Сера", "Кристаллы", "Самоцветы", "Золото"]
        return (icons[safe: object[5]] ?? "📦", names[safe: object[5]] ?? "Ресурс")
    default: return ("❓", "Объект типа \(object[4])")
    }
}

struct MapView: View {
    let header: Header
    let frame: Frame
    let fog: Bool
    let player: Int
    let selectObject: (Int?, Int?) -> Void
    @State private var tile = "Нажмите клетку: координаты и объекты"
    var shownObjects: [[Int]] {
        fog ? ReplayMapProjection.visibleObjects(frame.objects, ids:frame.players[String(player)]?.objects ?? []) : frame.objects
    }
    let colors = [0x6f5b43,0xc3af76,0x4b7743,0xd8e0ea,0x415844,0x786852,0x554d46,0x6e3127,0x2a5288,0x373b43]
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(player == 0 ? "ЗНАНИЕ КРАСНОГО ИГРОКА" : "ЗНАНИЕ СИНЕГО ИГРОКА").font(.caption.bold()).foregroundStyle(player == 0 ? red : blue)
            Canvas { context, size in
                let w = size.width / Double(header.width), h = size.height / Double(header.height)
                let visible = Array(frame.players[String(player)]?.visible ?? "")
                for i in header.terrain.indices {
                    let n = colors[safe: header.terrain[i]] ?? 0x505050
                    let color = Color(red: Double((n >> 16) & 255)/255, green: Double((n >> 8) & 255)/255, blue: Double(n & 255)/255)
                    let rect = CGRect(x: Double(i % header.width)*w, y: Double(i/header.width)*h, width: w-0.7, height: h-0.7)
                    context.fill(Path(rect), with: .color(fog && visible[safe:i] != "1" ? .black : color))
                }
                for obj in shownObjects.sorted(by: { ($0[4] == 34 ? 1 : 0) < ($1[4] == 34 ? 1 : 0) }) {
                    let shared = frame.objects.contains { $0[0] != obj[0] && $0[1] == obj[1] && $0[2] == obj[2] && (!fog || frame.players[String(player)]!.objects.contains($0[0])) }
                    let offset = shared ? (obj[4] == 34 ? 0.7 : -0.7) : 0.0
                    let x = (Double(obj[1])+0.5+offset)*w, y = (Double(obj[2])+0.5)*h
                    let side = min(w,h)*1.35
                    let rect = CGRect(x:x-side/2,y:y-side/2,width:side,height:side)
                    let color: Color = obj[6] == 1 ? blue : obj[6] == 0 ? red : .white.opacity(0.75)
                    let path = Path(roundedRect:rect,cornerRadius:side*0.2)
                    context.fill(path, with:.color(navy.opacity(0.92)))
                    context.stroke(path, with:.color(color),lineWidth:1.5)
                    let symbol = objectAppearance(obj).icon
                    context.draw(Text(symbol).font(.system(size:side*0.78,weight:.bold)).foregroundStyle(color), at:CGPoint(x:x,y:y))
                    if let owner = ReplayMapProjection.mineOwner(obj) {
                        let flag = CGRect(x:rect.maxX-side*0.48,y:rect.minY+1,width:side*0.44,height:max(3,side*0.18))
                        context.fill(Path(flag), with:.color(owner == 0 ? red : blue))
                        context.stroke(Path(flag), with:.color(navy),lineWidth:0.8)
                    }
                }
                for battle in frame.battles where !fog || frame.players[String(player)]?.visible.dropFirst(battle[1]*header.width+battle[0]).first == "1" {
                    context.draw(Text("⚔").font(.system(size: w*1.3)).foregroundStyle(.yellow), at: CGPoint(x:(Double(battle[0])+0.5)*w,y:(Double(battle[1])+0.5)*h))
                }
            }
            .aspectRatio(CGFloat(header.width)/CGFloat(header.height), contentMode: .fit)
            .overlay { GeometryReader { geo in Color.clear.contentShape(Rectangle()).onTapGesture { p in
                var x = min(header.width-1, Int(p.x/geo.size.width*Double(header.width)))
                var y = min(header.height-1, Int(p.y/geo.size.height*Double(header.height)))
                let shown = shownObjects
                let w = geo.size.width / Double(header.width), h = geo.size.height / Double(header.height)
                // Hit-test the displayed symbols, including co-located icon offsets.
                var picked: Int? = nil
                if let hit = shown.sorted(by: { ($0[4] == 34 ? 1 : 0) > ($1[4] == 34 ? 1 : 0) }).first(where: { obj in
                    let shared = shown.contains { $0[0] != obj[0] && $0[1] == obj[1] && $0[2] == obj[2] }
                    let offset = shared ? (obj[4] == 34 ? 0.7 : -0.7) : 0.0
                    return abs(p.x - (Double(obj[1])+0.5+offset)*w) <= min(w,h)*0.675 && abs(p.y - (Double(obj[2])+0.5)*h) <= min(w,h)*0.675
                }) { x = hit[1]; y = hit[2]; picked = hit[0] }
                let objects = shown.filter { $0[1] == x && $0[2] == y }
                selectObject(picked ?? objects.first?[0], fog ? player : nil)
                tile = "(\(x), \(y)) · " + (objects.isEmpty ? "нет видимых объектов" : objects.map { "\(objectAppearance($0).icon) \(objectAppearance($0).name) · id \($0[0])" }.joined(separator: "; "))
            } } }
            Text(tile).font(.caption2.monospaced()).foregroundStyle(.secondary).lineLimit(2).frame(height:28,alignment:.topLeading)
        }.padding(12).background(card).clipShape(RoundedRectangle(cornerRadius: 12))
    }
}
struct RawText: View {
    let text: String
    var body: some View {
        ScrollView([.vertical,.horizontal]) { Text(text).font(.system(size:11,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.topLeading).padding(12) }.background(navy)
    }
}
struct ContentView: View {
    @EnvironmentObject var store: Store
    @State private var selectedObject: Int?
    @State private var selectionPlayer: Int?
    var body: some View {
        VStack(spacing: 0) {
            if let r = store.replay, let e = store.event {
                header(r, e)
                HSplitView {
                    timeline.frame(minWidth:240,idealWidth:275,maxWidth:360)
                    ScrollView {
                        VStack(alignment:.leading,spacing:12) {
                            HStack(alignment:.top) { MapView(header:r.header,frame:r.frames[e.frame],fog:true,player:0,selectObject:selectObject); MapView(header:r.header,frame:r.frames[e.frame],fog:true,player:1,selectObject:selectObject) }
                            objectCard(r,e)
                            Text("♞ герой   🏰 город   ⛏️ шахта   🛡️ нейтралы   💰 золото   ⚔ автобой").font(.caption).foregroundStyle(.secondary)
                            army(e.army)
                            observationSummary(e.observation)
                            Text(e.title).font(.title2.bold()).foregroundStyle(e.actor.contains("Red") ? red : blue)
                            Text("\(e.actor) · \(e.source) · пакет \(e.packet)").font(.caption.monospaced())
                            Text(e.summary).font(.system(.body,design:.monospaced)).foregroundStyle(.yellow)
                            Text("Карта: последнее завершённое состояние движка #\(r.frames[e.frame].seq). Между состояниями показан предыдущий снимок. Observation — последние данные, доставленные сети; это не полная карта.").font(.caption).foregroundStyle(.secondary)
                        }.padding(16)
                    }.frame(minWidth:520,idealWidth:760)
                    inspector(e,r).frame(minWidth:280,idealWidth:350)
                }
                controls
            } else { VStack(spacing:20) { Text("Heroes Replay").font(.largeTitle); Button("Открыть replay…") { store.open() } }.frame(maxWidth:.infinity,maxHeight:.infinity) }
        }.background(navy).preferredColorScheme(.dark)
        .alert("Не удалось открыть replay", isPresented: Binding(get: { !store.error.isEmpty }, set: { if !$0 { store.error = "" } })) { Button("OK") { store.error = "" } } message: { Text(store.error) }
    }
    func selectObject(_ id: Int?, _ player: Int?) { selectedObject = id; selectionPlayer = player }
    func dictionary(_ text: String?) -> [String:Any]? {
        guard let data = text?.data(using:.utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with:data)) as? [String:Any]
    }
    @ViewBuilder func objectCard(_ r: Replay, _ e: Event) -> some View {
        if let id = selectedObject {
            let frame = r.frames[e.frame]
            if let obj = frame.objects.first(where:{$0[0] == id}), selectionPlayer == nil || frame.players[String(selectionPlayer!)]?.objects.contains(id) == true {
                let appearance = objectAppearance(obj)
                let meta = r.object_details?[String(id)]
                VStack(alignment:.leading,spacing:8) {
                    HStack {
                        Text("\(appearance.icon) \(obj[4] == 53 && meta?.subtype == "sawmill" ? "Лесопилка" : appearance.name)").font(.title3.bold())
                        Spacer()
                        Text("(\(obj[1]), \(obj[2])) · \(obj[6] == 1 ? "Синие" : obj[6] == 0 ? "Красные" : "Нейтральный")").font(.caption).foregroundStyle(.secondary)
                    }
                    if obj[4] == 79 {
                        Text("Ресурс на карте. Герой может подобрать его при посещении.").font(.callout)
                        if let quantity = meta?.quantity {
                            Text("\(appearance.name): \(quantity)").font(.headline).foregroundStyle(.yellow)
                            Text("Количество задано в исходной карте; после подбора объект исчезает.").font(.caption).foregroundStyle(.secondary)
                        } else { Text("Количество не записано.").font(.caption) }
                    } else if obj[4] == 34 {
                        Text("Герой с армией: перемещается по карте, посещает объекты и вступает в бой.").font(.callout)
                        if e.army.freshness == "unavailable" {
                            Text("Army and hero-window details are not included in this export.").font(.caption).foregroundStyle(.secondary)
                        } else if obj[6] == 1 && obj[5] == 2 {
                            Text("Армия: " + e.army.slots.map { "\($0.count) × \($0.creature == 0 ? "копейщики" : "тип \($0.creature)") (слот \($0.slot+1))" }.joined(separator:"; ")).font(.callout)
                            Text(e.army.freshness == "pre_battle" ? "Показана армия до автобоя; текущие потери не записаны." : "Последний снимок армии · пакет \(e.army.source_packet)").font(.caption).foregroundStyle(.secondary)
                            if let observation = dictionary(e.observation), let heroes = observation["heroes"] as? [[String:Any]], let hero = heroes.first(where: { $0["handle"] as? String == "object-0" }) {
                                Text("Очки хода: \(hero["movement"] as? Int ?? 0) / \(hero["movement_max"] as? Int ?? 0) · последнее наблюдение").font(.callout)
                            }
                            if let window = dictionary(e.hero_details) {
                                Text("Уровень: \(window["level"] as? Int ?? 0) · мана: \(window["mana"] as? Int ?? 0)").font(.callout)
                                if let skills = window["primary_skills"] as? [Int], skills.count == 4 {
                                    Text("Атака \(skills[0]) · защита \(skills[1]) · сила магии \(skills[2]) · знания \(skills[3])").font(.callout)
                                }
                                Text("Характеристики из последнего открытого окна · пакет \(e.hero_details_packet ?? 0). Артефакты и заклинания это окно не записывает.").font(.caption).foregroundStyle(.secondary)
                            }
                        } else {
                            Text("Текущая армия, мана, очки хода и инвентарь противника в этом журнале не записаны. Ниже — только сведения на старте.").font(.caption).foregroundStyle(.secondary)
                        }
                        if let meta { Text(meta.initial).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
                    } else if obj[4] == 54 {
                        Text("Нейтральный отряд, охраняющий проход. Посещение может начать бой.").font(.callout)
                        if let meta { Text(meta.initial).font(.headline) }
                        Text("Численность из исходной карты; это не счётчик потерь в бою.").font(.caption).foregroundStyle(.secondary)
                    } else if obj[4] == 53 {
                        Text(meta?.subtype == "sawmill" ? "Лесопилка — источник древесины. После захвата снабжает владельца ресурсом." : "Добывающий объект. После захвата снабжает владельца ресурсом.").font(.callout)
                    } else if obj[4] == 98 {
                        Text("Город: строительство зданий, найм войск и управление гарнизоном.").font(.callout)
                        if let meta { Text(meta.initial).font(.caption).foregroundStyle(.secondary) }
                        Text("Текущее содержимое открытого города доступно во вкладке «Вход»; стартовые здания выше не обновляются.").font(.caption).foregroundStyle(.secondary)
                    } else { Text("Дополнительное описание не записано.").font(.caption) }
                    Text("Карточка зрителя. Стартовые сведения из карты не являются входом нейросети.").font(.caption2).foregroundStyle(.secondary)
                }.padding(14).frame(maxWidth:.infinity,alignment:.leading).background(card).overlay(RoundedRectangle(cornerRadius:12).stroke(blue.opacity(0.5))).cornerRadius(12)
            } else {
                Text(selectionPlayer != nil ? "Выбранный объект сейчас не виден этому игроку или исчез с карты." : "Выбранный объект больше не присутствует на карте.").font(.callout).padding(14).frame(maxWidth:.infinity,alignment:.leading).background(card).cornerRadius(12)
            }
        } else {
            Text("Нажмите на иконку: здесь появятся название и описание объекта.").font(.callout).foregroundStyle(.secondary).padding(12)
        }
    }
    func header(_ r: Replay,_ e: Event) -> some View {
        HStack {
            VStack(alignment:.leading,spacing:4) { Text(r.title).font(.title2.bold()); Text(r.subtitle).font(.caption).foregroundStyle(.secondary) }
            Spacer()
            Text("ДЕНЬ \(r.frames[e.frame].day)").font(.headline).padding(10).background(card).cornerRadius(8)
            Text(r.frames[e.frame].boundary == "game_end" ? "ПОРАЖЕНИЕ СИНИХ" : "REPLAY VERIFIED").font(.caption.bold()).foregroundStyle(r.frames[e.frame].boundary == "game_end" ? red : .green)
            Button("Открыть…") { store.open() }
        }.padding(16)
    }
    var timeline: some View {
        VStack(alignment:.leading) {
            Picker("Шаги",selection:$store.mode) { Text("Действия").tag("Actions"); Text("Сеть").tag("Decisions"); Text("Всё").tag("All events") }.pickerStyle(.segmented).onChange(of:store.mode) { _, _ in
                if !store.filtered.contains(where:{$0.id == store.selected}), let next = store.filtered.first(where:{$0.id >= store.selected}) ?? store.filtered.last {store.selected=next.id}
            }
            TextField("Поиск события / игрока",text:$store.query).textFieldStyle(.roundedBorder).onChange(of:store.query) { _, _ in
                if !store.filtered.contains(where:{$0.id == store.selected}), let first = store.filtered.first {store.selected=first.id}
            }
            Text("\(store.filtered.count) событий · клик для перехода").font(.caption).foregroundStyle(.secondary)
            ScrollViewReader { proxy in
                List(selection: Binding<Int?>(get:{store.selected},set:{if let n=$0 {store.selected=n}})) {
                    ForEach(store.filtered) { e in
                        VStack(alignment:.leading,spacing:4) {
                            Text(e.title).font(.system(size:12,weight:.semibold)).lineLimit(2)
                            Text("\(e.actor) · \(e.source)").font(.system(size:10)).foregroundStyle(e.actor.contains("Red") ? red : .secondary).lineLimit(2)
                        }.padding(.vertical,3).tag(e.id).id(e.id)
                    }
                }.listStyle(.plain).onChange(of:store.selected) { _, n in proxy.scrollTo(n,anchor:.center) }
            }
        }.padding(10)
    }
    @ViewBuilder func observationSummary(_ text: String) -> some View {
        if let bytes = text.data(using:.utf8), let obj = (try? JSONSerialization.jsonObject(with:bytes)) as? [String:Any] {
            VStack(alignment:.leading,spacing:5) {
                Text("ПОСЛЕДНЕЕ НАБЛЮДЕНИЕ СЕТИ").font(.caption.bold()).foregroundStyle(blue)
                if let resources = obj["resources"] as? [Int], resources.count == 7 {
                    Text("Золото \(resources[6]) · дерево \(resources[0]) · руда \(resources[2]) · ртуть \(resources[1]) · сера \(resources[3]) · кристаллы \(resources[4]) · самоцветы \(resources[5])").font(.caption)
                }
                if let heroes = obj["heroes"] as? [[String:Any]] {
                    ForEach(Array(heroes.enumerated()),id:\.offset) { _, h in
                        Text("\(h["handle"] as? String ?? "Hero") · очки хода \(h["movement"] as? Int ?? 0) / \(h["movement_max"] as? Int ?? 0)").font(.caption)
                    }
                }
                let window = obj["window"] as? [String:Any]
                let context = window?["context"] as? [String:Any]
                Text("Окно: \(context?["kind"] as? String ?? "карта / нет открытого окна") · полная структура во вкладке «Вход»").font(.caption)
            }.padding(12).frame(maxWidth:.infinity,alignment:.leading).background(card).cornerRadius(12)
        }
    }
    func army(_ a: Army) -> some View {
        VStack(alignment:.leading,spacing:8) {
            if a.freshness == "unavailable" {
                Text("Army details are not included in this native-state export.").font(.caption).foregroundStyle(.secondary)
            } else {
            Text("\(a.name) · \(a.position.map { "(\($0[0]), \($0[1]))" } ?? "герой отсутствует") · \(a.slots.reduce(0){$0+$1.count}) существ").font(.headline)
            HStack { ForEach(0..<7) { i in
                let s = a.slots.first(where:{$0.slot == i})
                VStack { Text("Слот \(i+1)").font(.caption2).foregroundStyle(.secondary); Text(s.map { "\($0.count)" } ?? "—").font(.title3.bold()); Text(s.map { $0.creature == 0 ? "Копейщики" : "Тип \($0.creature)" } ?? "пусто").font(.system(size:10)).lineLimit(1).minimumScaleFactor(0.6) }.frame(maxWidth:.infinity).padding(8).background(navy).cornerRadius(6)
            } }
            Text(a.freshness == "pre_battle" ? "Армия перед автобоем; текущие потери в этом снимке не записаны." : "Последний снимок армии · пакет \(a.source_packet) · \(a.freshness)").font(.caption).foregroundStyle(.secondary)
            }
        }.padding(12).background(card).cornerRadius(12)
    }
    func inspector(_ e: Event,_ r: Replay) -> some View {
        VStack(alignment:.leading,spacing:8) {
            Text("ИНСПЕКТОР").font(.caption.bold())
            Picker("Данные",selection:$store.detail) { Text("Ивент").tag("Event"); Text("Вход").tag("Observation"); Text("Сеть").tag("Policy"); Text("Архив").tag("Archive") }.pickerStyle(.segmented)
            if store.detail == "Archive" {
                Picker("Файл",selection:$store.document) { ForEach(r.documents.keys.sorted(),id:\.self) { Text($0).tag($0) } }
                RawText(text:r.documents[store.document] ?? "").id(store.document)
            } else {
                Text(store.detail == "Policy" ? "Записанные log-probability, value и допустимые действия. Это не объяснение мыслей сети." : store.detail == "Observation" ? "Последнее доставленное наблюдение. Содержимое окна появляется только после открытия." : "Исходная запись без сокращений. Native input — команда и результат её применения.").font(.caption).foregroundStyle(.secondary)
                RawText(text:store.detail == "Observation" ? e.observation : store.detail == "Policy" ? (e.policy.isEmpty ? "На этом событии решения сети нет. Выберите шаг в режиме «Сеть»." : e.policy) : e.raw).id("\(e.id)-\(store.detail)")
            }
            Button("Скопировать JSON") {
                let text = store.detail == "Archive" ? r.documents[store.document] ?? "" : store.detail == "Observation" ? e.observation : store.detail == "Policy" ? e.policy : e.raw
                NSPasteboard.general.clearContents(); NSPasteboard.general.setString(text,forType:.string)
            }
        }.padding(12)
    }
    var controls: some View {
        HStack(spacing:16) {
            Button("⏮") { if let e=store.filtered.first {store.selected=e.id} }.help("Первое событие")
            Button("← Назад") {store.move(-1)}.keyboardShortcut(.leftArrow,modifiers:[])
            Button("Вперёд →") {store.move(1)}.keyboardShortcut(.rightArrow,modifiers:[])
            Button("⏭") {if let e=store.filtered.last {store.selected=e.id}}.help("Последнее событие")
            Slider(value:Binding(get:{Double(store.filtered.firstIndex(where:{$0.id == store.selected}) ?? 0)},set:{store.selected=store.filtered[min(Int($0),max(0,store.filtered.count-1))].id}),in:0...Double(max(1,store.filtered.count-1)),step:1).disabled(store.filtered.isEmpty)
            Text("\((store.filtered.firstIndex(where:{$0.id == store.selected}) ?? -1)+1) / \(store.filtered.count)").font(.caption.monospacedDigit()).frame(width:85)
        }.padding(14).background(card)
    }
}
@main struct HeroesReplayApp: App {
    @StateObject private var store = Store()
    var body: some Scene {
        WindowGroup("Heroes Replay") {
            ContentView().environmentObject(store).frame(minWidth:1150,minHeight:700)
                .onAppear {
                    if store.replay == nil {
                        if let arg = CommandLine.arguments.dropFirst().first, !arg.hasPrefix("-") {store.load(URL(fileURLWithPath:arg))}
                        else if let url=Bundle.main.url(forResource:"match",withExtension:"json") {store.load(url)}
                    }
                    NSApp.activate(ignoringOtherApps:true)
                }.onOpenURL {store.load($0)}
        }.defaultSize(width:1460,height:900)
        .commands { CommandGroup(replacing:.newItem) { Button("Открыть replay…") {store.open()}.keyboardShortcut("o") } }
    }
}
