// Semantic trees in native code: the trees heroes.model.policy builds from an observation, as a DAG with one node
// for each distinct content, encoded through fastnet's functions with a memo per node.
//
// policy.py stays the definition. Every tree here is the one policy._semantic / _observation_semantics builds
// (tests and tools compare the two), built from the same contract objects: the field plan of a dataclass is
// policy._field_plan's, the types kept by identity are policy._KEPT_TYPES. A tree's children are sorted by the
// repr of the Python tuple it stands for (policy._set), which is built here from the reprs Python gives the
// leaves; the UTF-8 bytes of two reprs order as their code points do, as Python's string order does.
//
// Numbers are told apart by their bits: 0.0 and -0.0 are two nodes, so each encodes from its own value, as
// fastnet does for the trees policy calls not exact (those never reach a memo there).
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

typedef long (*TextFn)(void *, const unsigned char *, long);
typedef long (*NumberFn)(void *, float, float);
typedef long (*NodeFn)(void *, long, const int *, long, int);

enum Kind : uint8_t { TEXT = 0, NUMBER = 1, NODE = 2, ORDERED = 3 };

struct Node {
    Kind kind;
    bool exact;
    int32_t tag = -1;          // the text node of a NODE's or ORDERED's tag
    double number = 0.0;       // NUMBER
    std::string text;          // TEXT, UTF-8
    std::string text_repr;     // TEXT: repr() of the str itself
    std::vector<int32_t> children;
    std::string repr;          // repr() of the tuple the node stands for, once asked
    bool has_repr = false;
    PyObject *python = nullptr;  // the Python tree, once asked (owned)
};

struct Held {  // a Python object kept alive while an entry refers to it by address
    std::unordered_map<PyObject *, int32_t> map;
    size_t limit;
    explicit Held(size_t limit) : limit(limit) {}
    int32_t find(PyObject *key) const {
        auto found = map.find(key);
        return found == map.end() ? -1 : found->second;
    }
    void put(PyObject *key, int32_t value) {
        if (map.size() >= limit) clear();
        Py_INCREF(key);
        map.emplace(key, value);
    }
    void clear() {
        for (auto &entry : map) Py_DECREF(entry.first);
        map.clear();
    }
};

struct Plan {
    struct Field { PyObject *name; int32_t tag; bool absent_skipped, category, amount, as_set; };
    std::vector<Field> fields;
    int32_t type_tag = -1;
    bool kept = false;
    bool dataclass = false;
};

// -- state of this process ---------------------------------------------------------------------------------------
std::vector<Node> nodes;
std::unordered_map<std::string, int32_t> texts;
std::unordered_map<uint64_t, int32_t> numbers;
std::unordered_map<std::string, int32_t> interior;  // kind, tag and children as bytes -> node
Held kept_values(65536), observations(8192);
std::unordered_map<PyTypeObject *, Plan> plans;
struct HandleIndex { PyObject *heroes = nullptr, *towns = nullptr, *objects = nullptr, *index = nullptr; } handles;

PyObject *field_plan, *kept_types, *kept_class;
PyTypeObject *recruitment_panel, *tavern_panel, *stack_panel, *dialog_panel, *dialog_choice, *notice_type;
PyObject *s_hashed, *s_exact, *s_text_attr, *s_kind, *s_world_object, *s_dataclass_fields, *s_name;
PyObject *s_window, *s_capability_panel, *s_items, *s_choices, *s_subject, *s_destination, *s_inspected_subject;
PyObject *s_size, *s_resources, *s_heroes, *s_towns, *s_objects, *s_context, *s_armies, *s_stack_panel;
PyObject *s_parent_context, *s_selection, *s_own_armies, *s_handle, *s_offers, *s_ref, *s_selected_offer_ref;
PyObject *s_selected_upgrade_ref, *s_stock, *s_unit_cost, *s_slider_max, *s_selected_detail, *s_gold_cost;
PyObject *s_recruit_status, *s_source, *s_total_cost, *s_affordable, *s_family, *s_components, *s_cancel;
PyObject *s_component_index, *s_display_mode, *s_exchange, *s_select_component;
TextFn text_fn;
NumberFn number_fn;
NodeFn node_fn;

// slots of nodes in one network's arena, valid for one epoch of it (FastPolicy._begin starts a new one)
void *slot_net = nullptr;
long slot_epoch = -1;
std::vector<int32_t> slots;

struct Fail {};  // a Python error is set

// -- nodes -------------------------------------------------------------------------------------------------------
int32_t text_node(const char *bytes, Py_ssize_t size) {
    std::string key(bytes, (size_t)size);
    auto found = texts.find(key);
    if (found != texts.end()) return found->second;
    PyObject *value = PyUnicode_DecodeUTF8(bytes, size, "strict");
    if (!value) throw Fail();
    PyObject *repr = PyObject_Repr(value);
    Py_DECREF(value);
    if (!repr) throw Fail();
    Py_ssize_t length;
    const char *chars = PyUnicode_AsUTF8AndSize(repr, &length);
    if (!chars) { Py_DECREF(repr); throw Fail(); }
    Node node;
    node.kind = TEXT;
    node.exact = true;
    node.text = key;
    node.text_repr.assign(chars, (size_t)length);
    Py_DECREF(repr);
    const int32_t index = (int32_t)nodes.size();
    nodes.push_back(std::move(node));
    texts.emplace(std::move(key), index);
    return index;
}

int32_t text_node(const char *ascii) { return text_node(ascii, (Py_ssize_t)strlen(ascii)); }

int32_t text_node(PyObject *unicode) {
    Py_ssize_t size;
    const char *bytes = PyUnicode_AsUTF8AndSize(unicode, &size);
    if (!bytes) throw Fail();
    return text_node(bytes, size);
}

int32_t number_node(double value) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof bits);
    auto found = numbers.find(bits);
    if (found != numbers.end()) return found->second;
    Node node;
    node.kind = NUMBER;
    node.number = value;
    node.exact = !(value != value) && !(value == 0.0 && std::signbit(value));
    const int32_t index = (int32_t)nodes.size();
    nodes.push_back(std::move(node));
    numbers.emplace(bits, index);
    return index;
}

int32_t interior_node(Kind kind, int32_t tag, const std::vector<int32_t> &children) {
    std::string key;
    key.reserve(5 + 4 * children.size());
    key.push_back((char)kind);
    key.append((const char *)&tag, 4);
    key.append((const char *)children.data(), 4 * children.size());
    auto found = interior.find(key);
    if (found != interior.end()) return found->second;
    Node node;
    node.kind = kind;
    node.tag = tag;
    node.children = children;
    node.exact = true;
    for (int32_t child : children) node.exact = node.exact && nodes[(size_t)child].exact;
    const int32_t index = (int32_t)nodes.size();
    nodes.push_back(std::move(node));
    interior.emplace(std::move(key), index);
    return index;
}

int32_t node(const char *tag, std::vector<int32_t> children = {}) { return interior_node(NODE, text_node(tag), children); }
int32_t node(int32_t tag, std::vector<int32_t> children = {}) { return interior_node(NODE, tag, children); }

const std::string &repr_of(int32_t index);

void float_repr(double value, std::string &out) {
    char *text = PyOS_double_to_string(value, 'r', 0, Py_DTSF_ADD_DOT_0, nullptr);
    if (!text) throw Fail();
    out += text;
    PyMem_Free(text);
}

const std::string &repr_of(int32_t index) {
    Node *item = &nodes[(size_t)index];
    if (item->has_repr) return item->repr;
    std::string out;
    switch (item->kind) {
        case TEXT: out = "('text', " + item->text_repr + ")"; break;
        case NUMBER: out = "('number', "; float_repr(item->number, out); out += ")"; break;
        default: {
            out = item->kind == NODE ? "('node', " : "('ordered', ";
            out += nodes[(size_t)item->tag].text_repr;
            out += ", (";
            const std::vector<int32_t> children = item->children;  // repr_of may grow the vector of nodes
            for (size_t i = 0; i < children.size(); i++) {
                if (i) out += ", ";
                out += repr_of(children[i]);
            }
            out += children.size() == 1 ? ",))" : "))";
        }
    }
    item = &nodes[(size_t)index];
    item->repr = std::move(out);
    item->has_repr = true;
    return item->repr;
}

int32_t sorted_set(int32_t tag, std::vector<int32_t> children) {
    for (int32_t child : children) repr_of(child);
    std::stable_sort(children.begin(), children.end(), [](int32_t a, int32_t b) {
        return nodes[(size_t)a].repr < nodes[(size_t)b].repr;
    });
    return interior_node(NODE, tag, children);
}

int32_t sorted_set(const char *tag, std::vector<int32_t> children) { return sorted_set(text_node(tag), std::move(children)); }

// -- Python objects ----------------------------------------------------------------------------------------------
struct Ref {  // an owned reference
    PyObject *p;
    explicit Ref(PyObject *p) : p(p) { if (!p) throw Fail(); }
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref &) = delete;
    Ref &operator=(const Ref &) = delete;
    PyObject *get() const { return p; }
};

PyObject *attr(PyObject *object, PyObject *name) {  // new reference
    PyObject *value = PyObject_GetAttr(object, name);
    if (!value) throw Fail();
    return value;
}

int32_t semantic(PyObject *value);

int32_t number_of(PyObject *value) {  // policy._number: ("number", float(value))
    if (PyFloat_CheckExact(value)) return number_node(PyFloat_AS_DOUBLE(value));
    const double number = PyFloat_AsDouble(value);
    if (number == -1.0 && PyErr_Occurred()) throw Fail();
    return number_node(number);
}

bool equal(PyObject *a, PyObject *b) {
    const int result = PyObject_RichCompareBool(a, b, Py_EQ);
    if (result < 0) throw Fail();
    return result;
}

int32_t truth(bool value) { return text_node(value ? "true" : "false"); }

std::vector<int32_t> each(PyObject *sequence, int32_t (*build)(PyObject *)) {
    Ref items(PySequence_Fast(sequence, "expected a sequence"));
    const Py_ssize_t count = PySequence_Fast_GET_SIZE(items.get());
    std::vector<int32_t> out;
    out.reserve((size_t)count);
    for (Py_ssize_t i = 0; i < count; i++) out.push_back(build(PySequence_Fast_GET_ITEM(items.get(), i)));
    return out;
}

Plan &plan_of(PyTypeObject *type) {
    auto found = plans.find(type);
    if (found != plans.end()) return found->second;
    Plan plan;
    const int kept = PySet_Contains(kept_types, (PyObject *)type);
    if (kept < 0) throw Fail();
    plan.kept = kept;
    plan.dataclass = PyObject_HasAttr((PyObject *)type, s_dataclass_fields);
    if (plan.dataclass) {
        Ref name(attr((PyObject *)type, s_name));
        plan.type_tag = text_node(name.get());
        Ref entries(PyObject_CallOneArg(field_plan, (PyObject *)type));
        Ref fast(PySequence_Fast(entries.get(), "field plan"));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast.get()); i++) {
            PyObject *entry = PySequence_Fast_GET_ITEM(fast.get(), i);
            Plan::Field field;
            field.name = PyTuple_GET_ITEM(entry, 0);
            Py_INCREF(field.name);
            PyUnicode_InternInPlace(&field.name);
            field.tag = text_node(field.name);
            field.absent_skipped = PyObject_IsTrue(PyTuple_GET_ITEM(entry, 1));
            field.category = PyObject_IsTrue(PyTuple_GET_ITEM(entry, 2));
            field.amount = PyObject_IsTrue(PyTuple_GET_ITEM(entry, 3));
            field.as_set = PyObject_IsTrue(PyTuple_GET_ITEM(entry, 4));
            plan.fields.push_back(field);
        }
    }
    return plans.emplace(type, std::move(plan)).first->second;
}

int32_t category(PyObject *item) {  // f"category:{item}" of an int
    Ref text(PyObject_Str(item));
    Py_ssize_t size;
    const char *chars = PyUnicode_AsUTF8AndSize(text.get(), &size);
    if (!chars) throw Fail();
    std::string out = "category:";
    out.append(chars, (size_t)size);
    return text_node(out.data(), (Py_ssize_t)out.size());
}

int32_t cards(PyObject *panel, PyObject *selected_name, const char *card_tag) {
    Ref offers(attr(panel, s_offers));
    Ref selected(attr(panel, selected_name));
    Ref items(PySequence_Fast(offers.get(), "offers"));
    std::vector<int32_t> out;
    const int32_t tag = text_node(card_tag);
    for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(items.get()); i++) {
        PyObject *card = PySequence_Fast_GET_ITEM(items.get(), i);
        Ref ref(attr(card, s_ref));
        out.push_back(node(tag, {semantic(card), truth(equal(ref.get(), selected.get()))}));
    }
    return sorted_set(strcmp(card_tag, "upgrade") == 0 ? "offers" : "cards", out);
}

int32_t dialog_choice_of(PyObject *panel, PyObject *choice) {  // policy._dialog_choice
    Ref kind(attr(choice, s_kind));
    int32_t component;
    if (equal(kind.get(), s_select_component)) {
        Ref components(attr(panel, s_components));
        Ref index(attr(choice, s_component_index));
        Ref item(PyObject_GetItem(components.get(), index.get()));
        component = semantic(item.get());
    } else {
        component = semantic(Py_None);
    }
    return node("dialog_choice", {text_node(kind.get()), component});
}

int32_t special(PyObject *value, PyTypeObject *type) {  // -1 when not one of policy._semantic's special types
    if (type == recruitment_panel) {
        int32_t set = cards(value, s_selected_offer_ref, "card");
        Ref stock(attr(value, s_stock)), cost(attr(value, s_unit_cost)), most(attr(value, s_slider_max)),
            destination(attr(value, s_destination));
        return node("recruitment", {set, node("stock", {number_of(stock.get())}), node("unit_cost", {semantic(cost.get())}),
                                    node("slider_max", {number_of(most.get())}), semantic(destination.get())});
    }
    if (type == tavern_panel) {
        int32_t set = cards(value, s_selected_offer_ref, "card");
        Ref detail(attr(value, s_selected_detail)), gold(attr(value, s_gold_cost)), status(attr(value, s_recruit_status));
        return node("tavern", {set, node("selected_detail", {semantic(detail.get())}), node("gold_cost", {number_of(gold.get())}),
                               node("recruit_status", {text_node(status.get())})});
    }
    if (type == stack_panel) {
        Ref parent(attr(value, s_parent_context)), source(attr(value, s_source));
        const int32_t first = semantic(parent.get()), second = semantic(source.get());
        int32_t set = cards(value, s_selected_upgrade_ref, "upgrade");
        Ref total(attr(value, s_total_cost)), affordable(attr(value, s_affordable));
        return node("stack_panel", {first, second, set, node("total_cost", {semantic(total.get())}),
                                    node("affordable", {semantic(affordable.get())})});
    }
    if (type == dialog_panel) {
        Ref family(attr(value, s_family)), text(attr(value, s_text_attr)), components(attr(value, s_components));
        const int32_t family_node = text_node(family.get()), text_tree = semantic(text.get());
        int32_t component_set = sorted_set("components", each(components.get(), semantic));
        Ref selection(attr(value, s_selection)), cancel(attr(value, s_cancel)), choices(attr(value, s_choices));
        const int32_t selection_node = node("selection", {semantic(selection.get())});
        const int32_t cancel_node = node("cancel", {semantic(cancel.get())});
        Ref items(PySequence_Fast(choices.get(), "choices"));
        std::vector<int32_t> out;
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(items.get()); i++)
            out.push_back(dialog_choice_of(value, PySequence_Fast_GET_ITEM(items.get(), i)));
        return node("dialog_panel", {family_node, text_tree, component_set, selection_node, cancel_node, sorted_set("choices", out)});
    }
    if (type == dialog_choice) {
        PyErr_SetString(PyExc_ValueError, "Dialog choice requires its displayed component binding");
        throw Fail();
    }
    if (type == notice_type) {
        Ref mode(attr(value, s_display_mode)), text(attr(value, s_text_attr)), components(attr(value, s_components));
        const int32_t mode_node = text_node(mode.get());
        const int32_t ordered = interior_node(ORDERED, text_node("notice_text"), each(text.get(), semantic));
        return node("notice", {mode_node, ordered, sorted_set("components", each(components.get(), semantic))});
    }
    return -1;
}

int32_t semantic(PyObject *value) {  // policy._semantic
    PyTypeObject *type = Py_TYPE(value);
    if (value == Py_None) return node("absent");
    if (PyBool_Check(value)) return truth(value == Py_True);
    if (PyFloat_CheckExact(value) || PyLong_CheckExact(value)) return number_of(value);
    if (PyUnicode_CheckExact(value)) return text_node(value);
    if (PyTuple_CheckExact(value)) return interior_node(ORDERED, text_node("tuple"), each(value, semantic));
    const int32_t found = special(value, type);
    if (found >= 0) return found;
    Plan &plan = plan_of(type);
    if (!plan.dataclass) {
        PyErr_SetString(PyExc_ValueError, "Unsupported model semantic payload");
        throw Fail();
    }
    if (plan.kept) {
        const int32_t known = kept_values.find(value);
        if (known >= 0) return known;
    }
    std::vector<int32_t> children;
    children.reserve(plan.fields.size());
    for (const Plan::Field &field : plan.fields) {
        Ref item(attr(value, field.name));
        PyObject *raw = item.get();
        if (field.absent_skipped && raw == Py_None) continue;
        int32_t encoded;
        if (field.category && PyLong_CheckExact(raw)) {
            encoded = category(raw);
        } else if (field.amount && PyLong_CheckExact(raw)) {
            Ref kind(attr(value, s_kind));
            encoded = equal(kind.get(), s_world_object) ? category(raw) : semantic(raw);
        } else if (field.as_set && PyTuple_CheckExact(raw)) {
            encoded = sorted_set(field.tag, each(raw, semantic));
        } else {
            encoded = semantic(raw);
        }
        children.push_back(node(field.tag, {encoded}));
    }
    const int32_t tree = node(plan.type_tag, children);
    if (plan.kept) kept_values.put(value, tree);
    return tree;
}

PyObject *handle_index(PyObject *observation) {  // borrowed: policy._handles
    Ref heroes(attr(observation, s_heroes)), towns(attr(observation, s_towns)), objects(attr(observation, s_objects));
    if (handles.index && heroes.get() == handles.heroes && towns.get() == handles.towns && objects.get() == handles.objects)
        return handles.index;
    PyObject *index = PyDict_New();
    if (!index) throw Fail();
    for (PyObject *group : {heroes.get(), towns.get(), objects.get()}) {
        Ref items(PySequence_Fast(group, "entries"));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(items.get()); i++) {
            PyObject *entry = PySequence_Fast_GET_ITEM(items.get(), i);
            PyObject *handle = PyObject_GetAttr(entry, s_handle);
            if (!handle) { Py_DECREF(index); throw Fail(); }
            PyObject *previous = PyDict_SetDefault(index, handle, entry);
            Py_DECREF(handle);
            if (!previous) { Py_DECREF(index); throw Fail(); }
        }
    }
    Py_XDECREF(handles.heroes); Py_XDECREF(handles.towns); Py_XDECREF(handles.objects); Py_XDECREF(handles.index);
    handles.heroes = heroes.get(); Py_INCREF(handles.heroes);
    handles.towns = towns.get(); Py_INCREF(handles.towns);
    handles.objects = objects.get(); Py_INCREF(handles.objects);
    handles.index = index;
    return index;
}

int32_t subject(PyObject *observation, PyObject *ref) {  // policy._subject
    if (ref == Py_None || observation == Py_None) return node("unresolved_reference");
    PyObject *entry = PyDict_GetItemWithError(handle_index(observation), ref);
    if (entry) return semantic(entry);
    if (PyErr_Occurred()) throw Fail();
    return node("unresolved_reference");
}

int32_t window_tree(PyObject *observation) {  // policy._window_semantic
    Ref window(attr(observation, s_window));
    const int32_t ordinary = semantic(window.get());
    if (window.get() == Py_None) return ordinary;
    Ref panel(attr(window.get(), s_capability_panel));
    if (panel.get() == Py_None) return ordinary;
    Ref items(attr(panel.get(), s_items)), choices(attr(panel.get(), s_choices));
    std::vector<int32_t> owned, bound;
    {
        Ref fast(PySequence_Fast(items.get(), "items"));
        const int32_t tag = text_node("owned_item");
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast.get()); i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(fast.get(), i);
            Ref owner(attr(item, s_subject));
            const int32_t first = semantic(item);
            owned.push_back(node(tag, {first, subject(observation, owner.get())}));
        }
    }
    {
        Ref fast(PySequence_Fast(choices.get(), "choices"));
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast.get()); i++) {
            PyObject *choice = PySequence_Fast_GET_ITEM(fast.get(), i);
            Ref actor(attr(choice, s_subject)), recipient(attr(choice, s_destination));
            const int32_t first = semantic(choice);
            const int32_t second = node("actor", {subject(observation, actor.get())});
            const int32_t third = recipient.get() != Py_None ? node("recipient", {subject(observation, recipient.get())})
                                                             : node("no_recipient");
            bound.push_back(node("capability_choice", {first, second, third}));
        }
    }
    const int32_t owned_set = sorted_set("bound_items", owned);
    const int32_t bound_set = sorted_set("bound_choices", bound);
    Ref inspected(attr(panel.get(), s_inspected_subject));
    return node("capability_window", {ordinary, owned_set, bound_set, node("inspected_object", {subject(observation, inspected.get())})});
}

int32_t observation_tree(PyObject *observation) {  // policy._observation_semantics
    const int32_t known = observations.find(observation);
    if (known >= 0) return known;
    Ref size(attr(observation, s_size)), resources(attr(observation, s_resources));
    Ref heroes(attr(observation, s_heroes)), towns(attr(observation, s_towns));
    const int32_t size_node = node("size", {semantic(size.get())});
    const int32_t resource_node = node("resources", {semantic(resources.get())});
    const int32_t hero_set = sorted_set("heroes", each(heroes.get(), semantic));
    const int32_t town_set = sorted_set("towns", each(towns.get(), semantic));
    const int32_t window_node = window_tree(observation);
    Ref context(attr(observation, s_context));
    Ref current(attr(context.get(), s_subject));
    const int32_t subject_node = node("current_window_subject", {subject(observation, current.get())});
    int32_t own = node("observation", {size_node, resource_node, hero_set, town_set, window_node, subject_node});
    Ref window(attr(observation, s_window));
    if (window.get() != Py_None) {
        Ref armies(attr(window.get(), s_armies));
        if (armies.get() != Py_None) {
            std::vector<int32_t> bindings;
            Ref fast(PySequence_Fast(armies.get(), "armies"));
            const int32_t tag = text_node("displayed_army");
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast.get()); i++) {
                PyObject *army = PySequence_Fast_GET_ITEM(fast.get(), i);
                Ref owner(attr(army, s_subject));
                const int32_t first = semantic(army);
                bindings.push_back(node(tag, {first, subject(observation, owner.get())}));
            }
            Ref stack(attr(window.get(), s_stack_panel));
            if (stack.get() != Py_None) {
                Ref parent(attr(stack.get(), s_parent_context));
                Ref parent_subject(attr(parent.get(), s_subject)), parent_kind(attr(parent.get(), s_kind));
                const int32_t first = subject(observation, parent_subject.get());
                int32_t second;
                if (equal(parent_kind.get(), s_exchange)) {
                    Ref selection(attr(parent.get(), s_selection));
                    second = subject(observation, selection.get());
                } else {
                    second = node("absent");
                }
                bindings.push_back(node("stack_parent", {first, second}));
            }
            Ref window_context(attr(window.get(), s_context));
            Ref kind(attr(window_context.get(), s_kind));
            if (equal(kind.get(), s_exchange)) {
                Ref selection(attr(window_context.get(), s_selection));
                bindings.push_back(node("exchange_partner", {subject(observation, selection.get())}));
            }
            own = node("management_observation", {own, sorted_set("army_bindings", bindings)});
        }
    }
    Ref armies(attr(observation, s_own_armies));
    if (armies.get() != Py_None) {
        std::vector<int32_t> bindings;
        Ref fast(PySequence_Fast(armies.get(), "own armies"));
        const int32_t tag = text_node("own_army");
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast.get()); i++) {
            PyObject *army = PySequence_Fast_GET_ITEM(fast.get(), i);
            Ref owner(attr(army, s_subject));
            const int32_t first = semantic(army);
            bindings.push_back(node(tag, {first, subject(observation, owner.get())}));
        }
        own = node("passive_army_observation", {own, sorted_set("own_army_bindings", bindings)});
    }
    observations.put(observation, own);
    return own;
}

// -- the Python tree of a node, and its encoding -----------------------------------------------------------------
PyObject *python_tree(int32_t index) {  // borrowed (owned by the node)
    Node *item = &nodes[(size_t)index];
    if (item->python) return item->python;
    PyObject *tree;
    if (item->kind == TEXT) {
        Ref value(PyUnicode_DecodeUTF8(item->text.data(), (Py_ssize_t)item->text.size(), "strict"));
        tree = Py_BuildValue("(sO)", "text", value.get());
        if (!tree) throw Fail();
    } else if (item->kind == NUMBER) {
        tree = Py_BuildValue("(sd)", "number", item->number);
        if (!tree) throw Fail();
    } else {
        const std::vector<int32_t> children = item->children;
        Ref tuple(PyTuple_New((Py_ssize_t)children.size()));
        for (size_t i = 0; i < children.size(); i++) {
            PyObject *child = python_tree(children[i]);
            Py_INCREF(child);
            PyTuple_SET_ITEM(tuple.get(), (Py_ssize_t)i, child);
        }
        item = &nodes[(size_t)index];
        const Node &tag = nodes[(size_t)item->tag];
        Ref tag_value(PyUnicode_DecodeUTF8(tag.text.data(), (Py_ssize_t)tag.text.size(), "strict"));
        Ref plain(Py_BuildValue("(sOO)", item->kind == NODE ? "node" : "ordered", tag_value.get(), tuple.get()));
        tree = PyObject_CallOneArg(kept_class, plain.get());  // policy._keep: a _Kept with its hash and exactness
        if (!tree) throw Fail();
        const Py_hash_t hash = PyObject_Hash(plain.get());
        if (hash == -1 && PyErr_Occurred()) { Py_DECREF(tree); throw Fail(); }
        Ref hashed(PyLong_FromSsize_t(hash));
        if (PyObject_SetAttr(tree, s_hashed, hashed.get()) || PyObject_SetAttr(tree, s_exact, item->exact ? Py_True : Py_False)) {
            Py_DECREF(tree);
            throw Fail();
        }
    }
    nodes[(size_t)index].python = tree;
    return tree;
}

long encode(void *net, int32_t index) {
    if ((size_t)index < slots.size() && slots[(size_t)index] >= 0) return slots[(size_t)index];
    const Node &item = nodes[(size_t)index];
    long slot;
    if (item.kind == TEXT) {
        slot = text_fn(net, (const unsigned char *)item.text.data(), (long)item.text.size());
    } else if (item.kind == NUMBER) {
        const double number = item.number;
        slot = number_fn(net, (float)std::copysign(std::log1p(std::fabs(number)), number), number == 0.0 ? 1.0f : 0.0f);
    } else {
        const std::vector<int32_t> children = item.children;
        const int32_t tag = item.tag;
        const bool ordered = item.kind == ORDERED;
        std::vector<int> child_slots(children.size());
        for (size_t i = 0; i < children.size(); i++) {
            const long child = encode(net, children[i]);
            if (child < 0) return child;
            child_slots[i] = (int)child;
        }
        const long tag_slot = encode(net, tag);
        if (tag_slot < 0) return tag_slot;
        slot = node_fn(net, tag_slot, child_slots.data(), (long)child_slots.size(), ordered ? 1 : 0);
    }
    if (slot < 0) {
        PyErr_SetString(PyExc_MemoryError, "The arena of tree encodings is full");
        throw Fail();
    }
    if ((size_t)index >= slots.size()) slots.resize(nodes.size() + 1024, -1);
    slots[(size_t)index] = (int32_t)slot;
    return slot;
}

// A long game meets millions of distinct trees; past this many nodes every table starts again (a miss only builds
// a tree again), and so do the memos that name nodes.
const size_t NODE_LIMIT = 4000000;

void reset_if_large() {
    if (nodes.size() < NODE_LIMIT) return;
    for (Node &item : nodes) Py_XDECREF(item.python);
    nodes.clear();
    texts.clear();
    numbers.clear();
    interior.clear();
    kept_values.clear();
    observations.clear();
    slots.clear();
    slot_net = nullptr;
}

PyObject *intern_name(const char *name) {
    PyObject *value = PyUnicode_InternFromString(name);
    if (!value) throw Fail();
    return value;
}

// -- the module --------------------------------------------------------------------------------------------------
PyObject *py_configure(PyObject *, PyObject *args) {
    PyObject *policy, *contract;
    unsigned long long text_address, number_address, node_address;
    if (!PyArg_ParseTuple(args, "OOKKK", &policy, &contract, &text_address, &number_address, &node_address)) return nullptr;
    try {
        field_plan = attr(policy, intern_name("_field_plan"));
        kept_types = attr(policy, intern_name("_KEPT_TYPES"));
        kept_class = attr(policy, intern_name("_Kept"));
        recruitment_panel = (PyTypeObject *)attr(contract, intern_name("RecruitmentPanel"));
        tavern_panel = (PyTypeObject *)attr(contract, intern_name("TavernPanel"));
        stack_panel = (PyTypeObject *)attr(contract, intern_name("StackPanel"));
        dialog_panel = (PyTypeObject *)attr(contract, intern_name("DialogPanel"));
        dialog_choice = (PyTypeObject *)attr(contract, intern_name("DialogChoice"));
        notice_type = (PyTypeObject *)attr(contract, intern_name("Notice"));
        s_hashed = intern_name("hashed"); s_exact = intern_name("exact"); s_text_attr = intern_name("text");
        s_kind = intern_name("kind"); s_world_object = intern_name("world_object");
        s_dataclass_fields = intern_name("__dataclass_fields__"); s_name = intern_name("__name__");
        s_window = intern_name("window"); s_capability_panel = intern_name("capability_panel");
        s_items = intern_name("items"); s_choices = intern_name("choices"); s_subject = intern_name("subject");
        s_destination = intern_name("destination"); s_inspected_subject = intern_name("inspected_subject");
        s_size = intern_name("size"); s_resources = intern_name("resources"); s_heroes = intern_name("heroes");
        s_towns = intern_name("towns"); s_objects = intern_name("objects"); s_context = intern_name("context");
        s_armies = intern_name("armies"); s_stack_panel = intern_name("stack_panel");
        s_parent_context = intern_name("parent_context"); s_selection = intern_name("selection");
        s_own_armies = intern_name("own_armies"); s_handle = intern_name("handle"); s_offers = intern_name("offers");
        s_ref = intern_name("ref"); s_selected_offer_ref = intern_name("selected_offer_ref");
        s_selected_upgrade_ref = intern_name("selected_upgrade_ref"); s_stock = intern_name("stock");
        s_unit_cost = intern_name("unit_cost"); s_slider_max = intern_name("slider_max");
        s_selected_detail = intern_name("selected_detail"); s_gold_cost = intern_name("gold_cost");
        s_recruit_status = intern_name("recruit_status"); s_source = intern_name("source");
        s_total_cost = intern_name("total_cost"); s_affordable = intern_name("affordable"); s_family = intern_name("family");
        s_components = intern_name("components"); s_cancel = intern_name("cancel");
        s_component_index = intern_name("component_index"); s_display_mode = intern_name("display_mode");
        s_exchange = intern_name("exchange"); s_select_component = intern_name("select_component");
    } catch (Fail &) {
        return nullptr;
    }
    text_fn = (TextFn)(uintptr_t)text_address;
    number_fn = (NumberFn)(uintptr_t)number_address;
    node_fn = (NodeFn)(uintptr_t)node_address;
    Py_RETURN_NONE;
}

PyObject *py_observation_tree(PyObject *, PyObject *observation) {
    try {
        reset_if_large();
        PyObject *tree = python_tree(observation_tree(observation));
        Py_INCREF(tree);
        return tree;
    } catch (Fail &) {
        return nullptr;
    }
}

PyObject *py_semantic_tree(PyObject *, PyObject *value) {
    try {
        reset_if_large();
        PyObject *tree = python_tree(semantic(value));
        Py_INCREF(tree);
        return tree;
    } catch (Fail &) {
        return nullptr;
    }
}

// observation_slot(net, epoch, observation): the arena slot of the observation's tree in that network
PyObject *py_observation_slot(PyObject *, PyObject *const *args, Py_ssize_t count) {
    if (count != 3) { PyErr_SetString(PyExc_TypeError, "observation_slot(net, epoch, observation)"); return nullptr; }
    void *net = PyLong_AsVoidPtr(args[0]);
    const long epoch = PyLong_AsLong(args[1]);
    if (PyErr_Occurred()) return nullptr;
    if (net != slot_net || epoch != slot_epoch) {
        slots.assign(slots.size(), -1);
        slot_net = net;
        slot_epoch = epoch;
    }
    try {
        reset_if_large();
        return PyLong_FromLong(encode(net, observation_tree(args[2])));
    } catch (Fail &) {
        return nullptr;
    }
}

PyObject *py_stats(PyObject *, PyObject *) {
    return Py_BuildValue("{s:n,s:n,s:n,s:n,s:n}", "nodes", (Py_ssize_t)nodes.size(), "texts", (Py_ssize_t)texts.size(),
                         "numbers", (Py_ssize_t)numbers.size(), "kept", (Py_ssize_t)kept_values.map.size(),
                         "observations", (Py_ssize_t)observations.map.size());
}

PyMethodDef methods[] = {
    {"configure", py_configure, METH_VARARGS, "configure(policy, contract, fn_text, fn_number, fn_node)"},
    {"observation_tree", py_observation_tree, METH_O, "policy.observation_semantics, from the native DAG"},
    {"semantic_tree", py_semantic_tree, METH_O, "policy._semantic, from the native DAG"},
    {"observation_slot", (PyCFunction)(void (*)(void))py_observation_slot, METH_FASTCALL,
     "observation_slot(net, epoch, observation): the arena slot of the observation's tree"},
    {"stats", py_stats, METH_NOARGS, "sizes of the native tables"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "heroes_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit_heroes_native(void) { return PyModule_Create(&module); }
