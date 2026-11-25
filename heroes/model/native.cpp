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
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <tuple>
#include <map>
#include <set>
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


// == contract values ==============================================================================================
// Every decoded contract value as one node per distinct content: equal values have one id, so comparing two
// observations, windows or armies is comparing two numbers. The schema is the contract's dataclasses, given by
// native.py at configure: each type's fields in order with their kinds, and the defaults of absent fields.

enum VKind : uint8_t { V_NONE = 0, V_BOOL = 1, V_INT = 2, V_STR = 3, V_TUPLE = 4, V_RECORD = 5 };

struct Val {
    VKind kind;
    int32_t type = -1;      // V_RECORD: the type's index
    int64_t number = 0;     // V_BOOL, V_INT
    int32_t string = -1;    // V_STR: the string's index
    std::vector<int32_t> items;  // V_TUPLE: elements; V_RECORD: fields in schema order
    PyObject *python = nullptr;  // the Python value, once asked (owned)
};

enum SKind : uint8_t { S_STR, S_INT, S_BOOL, S_OPTIONAL, S_TUPLE, S_FIXED, S_RECORD };

struct Spec {
    SKind kind;
    int32_t type = -1;            // S_RECORD
    std::vector<int32_t> inner;   // S_OPTIONAL, S_TUPLE: one spec; S_FIXED: one per element (indices into specs)
};

struct VField {
    std::string name;
    PyObject *python_name = nullptr;  // interned
    int32_t spec;
    int32_t fallback = -1;            // value id of the default, -1 when the field is required
};

struct VType {
    std::string name;
    PyObject *python = nullptr;   // the dataclass
    std::vector<VField> fields;
    std::unordered_map<std::string, int32_t> by_name;
};

std::vector<Val> vals;
std::unordered_map<std::string, int32_t> val_index;   // kind, type, scalar and items as bytes -> value
std::vector<std::string> strings;
std::vector<PyObject *> string_objects;               // the Python str of each string, once asked
std::unordered_map<std::string, int32_t> string_index;
std::vector<Spec> specs;
std::vector<VType> vtypes;
std::unordered_map<std::string, int32_t> vtype_index;
int32_t v_none = -1, v_true = -1, v_false = -1, v_empty = -1;

int32_t string_id(const char *bytes, size_t size) {
    std::string key(bytes, size);
    auto found = string_index.find(key);
    if (found != string_index.end()) return found->second;
    const int32_t index = (int32_t)strings.size();
    strings.push_back(key);
    string_objects.push_back(nullptr);
    string_index.emplace(std::move(key), index);
    return index;
}

int32_t intern_val(Val &&value) {
    std::string key;
    key.reserve(16 + 4 * value.items.size());
    key.push_back((char)value.kind);
    key.append((const char *)&value.type, 4);
    key.append((const char *)&value.number, 8);
    key.append((const char *)&value.string, 4);
    key.append((const char *)value.items.data(), 4 * value.items.size());
    auto found = val_index.find(key);
    if (found != val_index.end()) return found->second;
    const int32_t index = (int32_t)vals.size();
    vals.push_back(std::move(value));
    val_index.emplace(std::move(key), index);
    return index;
}

int32_t v_scalar(VKind kind, int64_t number, int32_t string = -1) {
    Val value;
    value.kind = kind;
    value.number = number;
    value.string = string;
    return intern_val(std::move(value));
}

int32_t v_items(VKind kind, int32_t type, std::vector<int32_t> items) {
    Val value;
    value.kind = kind;
    value.type = type;
    value.items = std::move(items);
    return intern_val(std::move(value));
}

struct JsonError {};

struct Json {  // a strict JSON reader over one message, decoding by the contract's specs
    const char *p, *end;
    std::string scratch;

    void fail(const char *what) {
        if (!PyErr_Occurred()) PyErr_Format(PyExc_ValueError, "Engine message: %s", what);
        throw JsonError();
    }
    void ws() { while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++; }
    bool take(char c) { ws(); if (p < end && *p == c) { p++; return true; } return false; }
    void expect(char c) { if (!take(c)) fail("unexpected character"); }
    bool literal(const char *word) {
        ws();
        const size_t n = strlen(word);
        if ((size_t)(end - p) >= n && memcmp(p, word, n) == 0) { p += n; return true; }
        return false;
    }
    static void utf8(std::string &out, uint32_t code) {
        if (code < 0x80) out.push_back((char)code);
        else if (code < 0x800) { out.push_back((char)(0xC0 | (code >> 6))); out.push_back((char)(0x80 | (code & 0x3F))); }
        else if (code < 0x10000) { out.push_back((char)(0xE0 | (code >> 12))); out.push_back((char)(0x80 | ((code >> 6) & 0x3F)));
                                   out.push_back((char)(0x80 | (code & 0x3F))); }
        else { out.push_back((char)(0xF0 | (code >> 18))); out.push_back((char)(0x80 | ((code >> 12) & 0x3F)));
               out.push_back((char)(0x80 | ((code >> 6) & 0x3F))); out.push_back((char)(0x80 | (code & 0x3F))); }
    }
    uint32_t hex4() {
        if (end - p < 4) fail("short escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; i++) {
            const char c = *p++;
            value <<= 4;
            if (c >= '0' && c <= '9') value |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') value |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= (uint32_t)(c - 'A' + 10);
            else fail("bad escape");
        }
        return value;
    }
    const std::string &string() {  // into scratch, UTF-8
        ws();
        if (p >= end || *p != '"') fail("expected a string");
        p++;
        scratch.clear();
        {   // the common case: no escape before the closing quote, copied at once
            const char *quote = (const char *)memchr(p, '"', (size_t)(end - p));
            if (quote && !memchr(p, '\\', (size_t)(quote - p))) {
                for (const char *c = p; c < quote; c++) if ((unsigned char)*c < 0x20) fail("control character in string");
                scratch.assign(p, (size_t)(quote - p));
                p = quote + 1;
                return scratch;
            }
        }
        while (true) {
            if (p >= end) fail("unterminated string");
            const char c = *p++;
            if (c == '"') break;
            if ((unsigned char)c < 0x20) fail("control character in string");
            if (c != '\\') { scratch.push_back(c); continue; }
            if (p >= end) fail("bad escape");
            const char e = *p++;
            switch (e) {
                case '"': scratch.push_back('"'); break;
                case '\\': scratch.push_back('\\'); break;
                case '/': scratch.push_back('/'); break;
                case 'b': scratch.push_back('\b'); break;
                case 'f': scratch.push_back('\f'); break;
                case 'n': scratch.push_back('\n'); break;
                case 'r': scratch.push_back('\r'); break;
                case 't': scratch.push_back('\t'); break;
                case 'u': {
                    uint32_t code = hex4();
                    if (code >= 0xD800 && code < 0xDC00 && end - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        const char *mark = p;
                        p += 2;
                        const uint32_t low = hex4();
                        if (low >= 0xDC00 && low < 0xE000) code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                        else p = mark;
                    }
                    if (code >= 0xD800 && code < 0xE000) fail("lone surrogate");  // json would keep it; no engine sends one
                    utf8(scratch, code);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return scratch;
    }
    int64_t integer() {
        ws();
        const char *start = p;
        if (p < end && *p == '-') p++;
        if (p >= end || *p < '0' || *p > '9') fail("expected an integer");
        if (*p == '0' && p + 1 < end && p[1] >= '0' && p[1] <= '9') fail("leading zero");
        while (p < end && *p >= '0' && *p <= '9') p++;
        if (p < end && (*p == '.' || *p == 'e' || *p == 'E')) fail("expected an integer, not a float");
        if (p - start > 18) fail("integer out of range");
        return std::strtoll(std::string(start, (size_t)(p - start)).c_str(), nullptr, 10);
    }
    void skip_string() {  // a string, not kept
        if (p >= end || *p != '"') fail("expected a string");
        p++;
        while (true) {
            const char *quote = (const char *)memchr(p, '"', (size_t)(end - p));
            if (!quote) fail("unterminated string");
            const char *back = quote;  // a quote preceded by an odd number of backslashes is escaped
            while (back > p && back[-1] == '\\') back--;
            p = quote + 1;
            if (((quote - back) & 1) == 0) return;
        }
    }
    void skip() {  // any value
        ws();
        if (p >= end) fail("unexpected end");
        const char c = *p;
        if (c == '{') { p++; if (take('}')) return; do { ws(); skip_string(); expect(':'); skip(); } while (take(',')); expect('}'); }
        else if (c == '[') { p++; if (take(']')) return; do { skip(); } while (take(',')); expect(']'); }
        else if (c == '"') skip_string();
        else if (literal("true") || literal("false") || literal("null")) return;
        else { if (*p == '-') p++; while (p < end && (std::isdigit((unsigned char)*p) || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) p++; }
    }

    int32_t value(int32_t spec_index) {
        const Spec &spec = specs[(size_t)spec_index];
        switch (spec.kind) {
            case S_OPTIONAL: if (literal("null")) return v_none; return value(spec.inner[0]);
            case S_STR: { const std::string &text = string(); return v_scalar(V_STR, 0, string_id(text.data(), text.size())); }
            case S_INT: return v_scalar(V_INT, integer());
            case S_BOOL: if (literal("true")) return v_true; if (literal("false")) return v_false; fail("expected a boolean");
            case S_TUPLE: {
                expect('[');
                std::vector<int32_t> items;
                if (!take(']')) { do items.push_back(value(spec.inner[0])); while (take(',')); expect(']'); }
                return items.empty() ? v_empty : v_items(V_TUPLE, -1, std::move(items));
            }
            case S_FIXED: {
                expect('[');
                std::vector<int32_t> items;
                for (size_t i = 0; i < spec.inner.size(); i++) { if (i) expect(','); items.push_back(value(spec.inner[i])); }
                expect(']');
                return v_items(V_TUPLE, -1, std::move(items));
            }
            case S_RECORD: return record(spec.type);
        }
        fail("bad spec");
        return -1;
    }

    int32_t record(int32_t type_index) {
        const VType &type = vtypes[(size_t)type_index];
        std::vector<int32_t> items(type.fields.size(), -1);
        expect('{');
        if (!take('}')) {
            do {
                const std::string &key = string();
                auto found = type.by_name.find(key);
                if (found == type.by_name.end()) {
                    PyErr_Format(PyExc_ValueError, "Unknown fields in %s", type.name.c_str());
                    fail("unknown field");
                }
                expect(':');
                items[(size_t)found->second] = value(type.fields[(size_t)found->second].spec);
            } while (take(','));
            expect('}');
        }
        for (size_t i = 0; i < items.size(); i++) {
            if (items[i] >= 0) continue;
            if (type.fields[i].fallback < 0) {
                PyErr_Format(PyExc_ValueError, "Missing or invalid %s fields", type.name.c_str());
                fail("missing field");
            }
            items[i] = type.fields[i].fallback;
        }
        return v_items(V_RECORD, type_index, std::move(items));
    }
};

PyObject *string_object(int32_t index) {  // borrowed
    PyObject *&slot = string_objects[(size_t)index];
    if (!slot) {
        slot = PyUnicode_DecodeUTF8(strings[(size_t)index].data(), (Py_ssize_t)strings[(size_t)index].size(), "strict");
        if (!slot) throw Fail();
        PyUnicode_InternInPlace(&slot);
    }
    return slot;
}

PyObject *python_value(int32_t index) {  // borrowed: the contract object this value stands for
    Val *value = &vals[(size_t)index];
    if (value->python) return value->python;
    PyObject *made = nullptr;
    switch (value->kind) {
        case V_NONE: made = Py_None; Py_INCREF(made); break;
        case V_BOOL: made = value->number ? Py_True : Py_False; Py_INCREF(made); break;
        case V_INT: made = PyLong_FromLongLong(value->number); break;
        case V_STR: made = string_object(value->string); Py_INCREF(made); break;
        case V_TUPLE: {
            const std::vector<int32_t> items = value->items;
            made = PyTuple_New((Py_ssize_t)items.size());
            if (!made) throw Fail();
            for (size_t i = 0; i < items.size(); i++) {
                PyObject *item = python_value(items[i]);
                Py_INCREF(item);
                PyTuple_SET_ITEM(made, (Py_ssize_t)i, item);
            }
            break;
        }
        case V_RECORD: {
            const std::vector<int32_t> items = value->items;
            const VType &type = vtypes[(size_t)value->type];
            Ref arguments(PyTuple_New(0));
            Ref keywords(PyDict_New());
            for (size_t i = 0; i < items.size(); i++)
                if (PyDict_SetItem(keywords.get(), type.fields[i].python_name, python_value(items[i]))) throw Fail();
            made = PyObject_Call(type.python, arguments.get(), keywords.get());
            break;
        }
    }
    if (!made) throw Fail();
    vals[(size_t)index].python = made;
    return made;
}

// configure_values(types, specs): types = [(name, dataclass, [(field, spec index, default or a marker)])],
// specs = [(kind, type index, [inner spec indices])]; the defaults are Python values of the schema.
int32_t value_of_python(PyObject *object, int32_t spec_index);

int32_t value_of_python(PyObject *object, int32_t spec_index) {
    const Spec spec = specs[(size_t)spec_index];
    switch (spec.kind) {
        case S_OPTIONAL: return object == Py_None ? v_none : value_of_python(object, spec.inner[0]);
        case S_STR: { Py_ssize_t size; const char *chars = PyUnicode_AsUTF8AndSize(object, &size); if (!chars) throw Fail();
                      return v_scalar(V_STR, 0, string_id(chars, (size_t)size)); }
        case S_INT: { const long long number = PyLong_AsLongLong(object); if (number == -1 && PyErr_Occurred()) throw Fail();
                      return v_scalar(V_INT, number); }
        case S_BOOL: return object == Py_True ? v_true : v_false;
        case S_TUPLE: case S_FIXED: {
            Ref fast(PySequence_Fast(object, "tuple"));
            std::vector<int32_t> items;
            for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fast.get()); i++)
                items.push_back(value_of_python(PySequence_Fast_GET_ITEM(fast.get(), i),
                                                spec.kind == S_TUPLE ? spec.inner[0] : spec.inner[(size_t)i]));
            return spec.kind == S_TUPLE && items.empty() ? v_empty : v_items(V_TUPLE, -1, std::move(items));
        }
        case S_RECORD: {
            const VType &type = vtypes[(size_t)spec.type];
            std::vector<int32_t> items;
            for (const VField &field : type.fields) {
                Ref item(attr(object, field.python_name));
                items.push_back(value_of_python(item.get(), field.spec));
            }
            return v_items(V_RECORD, spec.type, std::move(items));
        }
    }
    return -1;
}

PyObject *py_configure_values(PyObject *, PyObject *args) {
    PyObject *type_list, *spec_list, *missing;
    if (!PyArg_ParseTuple(args, "OOO", &type_list, &spec_list, &missing)) return nullptr;
    try {
        specs.clear();
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(spec_list); i++) {
            PyObject *entry = PyList_GET_ITEM(spec_list, i);
            Spec spec;
            spec.kind = (SKind)PyLong_AsLong(PyTuple_GET_ITEM(entry, 0));
            spec.type = (int32_t)PyLong_AsLong(PyTuple_GET_ITEM(entry, 1));
            PyObject *inner = PyTuple_GET_ITEM(entry, 2);
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(inner); j++) spec.inner.push_back((int32_t)PyLong_AsLong(PyList_GET_ITEM(inner, j)));
            specs.push_back(spec);
        }
        vtypes.clear();
        vtype_index.clear();
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(type_list); i++) {
            PyObject *entry = PyList_GET_ITEM(type_list, i);
            VType type;
            Py_ssize_t size;
            const char *name = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(entry, 0), &size);
            type.name.assign(name, (size_t)size);
            type.python = PyTuple_GET_ITEM(entry, 1);
            Py_INCREF(type.python);
            vtype_index[type.name] = (int32_t)i;
            vtypes.push_back(std::move(type));
        }
        v_none = v_scalar(V_NONE, 0);
        v_true = v_scalar(V_BOOL, 1);
        v_false = v_scalar(V_BOOL, 0);
        v_empty = v_items(V_TUPLE, -1, {});
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(type_list); i++) {  // fields after every type exists (defaults)
            PyObject *fields = PyTuple_GET_ITEM(PyList_GET_ITEM(type_list, i), 2);
            VType &type = vtypes[(size_t)i];
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(fields); j++) {
                PyObject *field_entry = PyList_GET_ITEM(fields, j);
                VField field;
                Py_ssize_t size;
                const char *name = PyUnicode_AsUTF8AndSize(PyTuple_GET_ITEM(field_entry, 0), &size);
                field.name.assign(name, (size_t)size);
                field.python_name = PyTuple_GET_ITEM(field_entry, 0);
                Py_INCREF(field.python_name);
                PyUnicode_InternInPlace(&field.python_name);
                field.spec = (int32_t)PyLong_AsLong(PyTuple_GET_ITEM(field_entry, 1));
                type.by_name[field.name] = (int32_t)type.fields.size();
                type.fields.push_back(field);
            }
        }
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(type_list); i++) {
            PyObject *fields = PyTuple_GET_ITEM(PyList_GET_ITEM(type_list, i), 2);
            for (Py_ssize_t j = 0; j < PyList_GET_SIZE(fields); j++) {
                PyObject *fallback = PyTuple_GET_ITEM(PyList_GET_ITEM(fields, j), 2);
                if (fallback != missing)
                    vtypes[(size_t)i].fields[(size_t)j].fallback = value_of_python(fallback, vtypes[(size_t)i].fields[(size_t)j].spec);
            }
        }
    } catch (Fail &) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

// decode(bytes, type name): the value of one JSON document of that contract type (tests)
PyObject *py_decode(PyObject *, PyObject *args) {
    Py_buffer buffer;
    const char *type_name;
    if (!PyArg_ParseTuple(args, "y*s", &buffer, &type_name)) return nullptr;
    auto found = vtype_index.find(type_name);
    if (found == vtype_index.end()) { PyBuffer_Release(&buffer); PyErr_SetString(PyExc_KeyError, type_name); return nullptr; }
    Json reader{(const char *)buffer.buf, (const char *)buffer.buf + buffer.len, {}};
    int32_t result;
    try {
        result = reader.record(found->second);
        reader.ws();
        if (reader.p != reader.end) reader.fail("trailing data");
    } catch (JsonError &) {
        PyBuffer_Release(&buffer);
        return nullptr;
    } catch (Fail &) {
        PyBuffer_Release(&buffer);
        return nullptr;
    }
    PyBuffer_Release(&buffer);
    return PyLong_FromLong(result);
}

PyObject *py_python_value(PyObject *, PyObject *index) {
    const long value = PyLong_AsLong(index);
    if (value == -1 && PyErr_Occurred()) return nullptr;
    if (value < 0 || (size_t)value >= vals.size()) { PyErr_SetString(PyExc_IndexError, "no such value"); return nullptr; }
    try {
        PyObject *made = python_value((int32_t)value);
        Py_INCREF(made);
        return made;
    } catch (Fail &) {
        return nullptr;
    }
}


// == the producer: one seat's stream of engine messages as frames ==================================================
// protocol.Producer.ingest, for the training interface: each record of a contextual message becomes a frame of the
// contract's Frame shape, its payloads values of the contract's types (decision ids made frame numbers, a decision
// without an observation holding its view's, a view without its map holding the previous view's, tile changes laid
// on the previous tiles). Its frames turn into contract Frames for the Python path and for checks.

enum FKind : int8_t { F_DECISION, F_VIEW, F_EVENT, F_CONSUMED, F_ISSUED, F_ACKNOWLEDGED, F_RESULT, F_CONTEXT,
                      F_CONTEXT_INVALIDATED, F_CONTEXT_OPENED, F_SUPERSEDED, F_TERMINAL, F_SUBMITTED, F_KINDS };
const char *KIND_NAMES[F_KINDS] = {"decision", "view", "event", "consumed", "issued", "acknowledged", "result", "context",
                                   "context_invalidated", "context_opened", "superseded", "terminal", "submitted"};

struct NFrame {
    FKind kind;
    int32_t seq = 0, day = 0;
    int32_t observation = -1, context = -1, offers = -1, action = -1, result = -1, event = -1;  // values
    int32_t decision_seq = -1;
    int64_t revision = -1, command_index = -1;
    bool has_revision = false, has_command = false;
    int32_t consumer = -1, reason = -1, outcome = -1;  // strings
    int8_t accepted = -1;
};

struct Producer {
    PyObject_HEAD
    std::vector<NFrame> *frames;
    std::unordered_map<int64_t, int32_t> *decision_ids;
    std::string *episode;
    int64_t player = 0, protocol = 0, current_id = -1;
    int32_t view = -1, terminal = -1, gold_mine_captures = 0;
    bool ready = false, has_current = false;
    std::string *interface, *history_schema, *action_schema, *army_information;
    PyObject *python_frames;  // list: contract Frames made so far (frames before ``made`` are converted)
};

int32_t type_named(const char *name) {
    auto found = vtype_index.find(name);
    if (found == vtype_index.end()) { PyErr_Format(PyExc_KeyError, "contract type %s", name); throw Fail(); }
    return found->second;
}

int32_t field_of(int32_t type, const char *name) {
    const VType &found = vtypes[(size_t)type];
    auto entry = found.by_name.find(name);
    if (entry == found.by_name.end()) { PyErr_Format(PyExc_KeyError, "field %s.%s", found.name.c_str(), name); throw Fail(); }
    return entry->second;
}

struct Types {  // indices of the types and fields the producer reads
    int32_t observation, offers, context, action, result, event, window, hero, town, tile;
    int32_t o_size, o_tiles, o_objects, o_heroes, o_towns, o_window, o_own_armies;
    int32_t w_context, w_town_panel, w_armies, w_stack_panel, w_dialog_panel, w_capability_controls, w_capability_panel;
    int32_t c_kind, h_handle, h_in_garrison, t_handle, tile_position;
    int32_t event_kind, event_notice;
    bool ready = false;
} T;

void prepare_types() {
    if (T.ready) return;
    T.observation = type_named("Observation"); T.offers = type_named("Offers"); T.context = type_named("Context");
    T.action = type_named("Action"); T.result = type_named("Result"); T.event = type_named("VisibleEvent");
    T.window = type_named("Window"); T.hero = type_named("Hero"); T.town = type_named("Town"); T.tile = type_named("Tile");
    T.o_size = field_of(T.observation, "size"); T.o_tiles = field_of(T.observation, "tiles");
    T.o_objects = field_of(T.observation, "objects"); T.o_heroes = field_of(T.observation, "heroes");
    T.o_towns = field_of(T.observation, "towns"); T.o_window = field_of(T.observation, "window");
    T.o_own_armies = field_of(T.observation, "own_armies");
    T.w_context = field_of(T.window, "context"); T.w_town_panel = field_of(T.window, "town_panel");
    T.w_armies = field_of(T.window, "armies"); T.w_stack_panel = field_of(T.window, "stack_panel");
    T.w_dialog_panel = field_of(T.window, "dialog_panel"); T.w_capability_controls = field_of(T.window, "capability_controls");
    T.w_capability_panel = field_of(T.window, "capability_panel");
    T.c_kind = field_of(T.context, "kind"); T.h_handle = field_of(T.hero, "handle"); T.h_in_garrison = field_of(T.hero, "in_garrison");
    T.t_handle = field_of(T.town, "handle"); T.tile_position = field_of(T.tile, "position");
    T.event_kind = field_of(T.event, "kind"); T.event_notice = field_of(T.event, "notice");
    T.ready = true;
}

inline const Val &V(int32_t index) { return vals[(size_t)index]; }
inline int32_t field(int32_t value, int32_t index) { return V(value).items[(size_t)index]; }
inline bool str_is(int32_t value, const char *text) {
    const Val &item = V(value);
    return item.kind == V_STR && strings[(size_t)item.string] == text;
}

int32_t default_context() {  // Context()
    static int32_t made = -1;
    if (made < 0) {
        std::vector<int32_t> items;
        for (const VField &item : vtypes[(size_t)T.context].fields) items.push_back(item.fallback);
        made = v_items(V_RECORD, T.context, items);
    }
    return made;
}

int32_t record_of(int32_t type, std::vector<std::pair<const char *, int32_t>> given) {
    const VType &found = vtypes[(size_t)type];
    std::vector<int32_t> items;
    for (const VField &item : found.fields) items.push_back(item.fallback);
    for (auto &entry : given) items[(size_t)field_of(type, entry.first)] = entry.second;
    for (int32_t item : items) if (item < 0) { PyErr_Format(PyExc_ValueError, "Missing %s fields", found.name.c_str()); throw Fail(); }
    return v_items(V_RECORD, type, items);
}

int32_t str_value(const std::string &text) { return v_scalar(V_STR, 0, string_id(text.data(), text.size())); }

int32_t offers_of(int32_t observation, const std::string &action_schema) {  // Offers.from_observation (v8 profile)
    static std::unordered_map<int64_t, int32_t> built;
    const int64_t key = ((int64_t)observation << 20) ^ (int64_t)string_id(action_schema.data(), action_schema.size());
    auto found = built.find(key);
    if (found != built.end()) return found->second;
    const int32_t window = field(observation, T.o_window);
    const bool windowed = V(window).kind == V_RECORD;
    const int32_t context = windowed ? field(window, T.w_context) : default_context();
    const bool map = str_is(field(context, T.c_kind), "map");
    std::vector<int32_t> movable;
    if (map)
        for (int32_t hero : V(field(observation, T.o_heroes)).items)
            if (V(field(hero, T.h_in_garrison)).kind != V_BOOL || !V(field(hero, T.h_in_garrison)).number) movable.push_back(field(hero, T.h_handle));
    std::vector<std::pair<const char *, int32_t>> given = {
        {"context", context}, {"size", field(observation, T.o_size)}, {"schema", str_value(action_schema)},
        {"armies", windowed ? field(window, T.w_armies) : v_empty},
        {"stack_panel", windowed ? field(window, T.w_stack_panel) : v_none},
        {"movable_hero_refs", movable.empty() ? v_empty : v_items(V_TUPLE, -1, movable)},
        {"dialog_panel", windowed ? field(window, T.w_dialog_panel) : v_none},
        {"capability_controls", windowed ? field(window, T.w_capability_controls) : v_empty},
        {"capability_panel", windowed ? field(window, T.w_capability_panel) : v_none}};
    if (!map) {
        given.push_back({"panel", field(window, T.w_town_panel)});
    } else {
        std::vector<int32_t> heroes, towns;
        for (int32_t hero : V(field(observation, T.o_heroes)).items) heroes.push_back(field(hero, T.h_handle));
        for (int32_t town : V(field(observation, T.o_towns)).items) towns.push_back(field(town, T.t_handle));
        given.push_back({"hero_refs", heroes.empty() ? v_empty : v_items(V_TUPLE, -1, heroes)});
        given.push_back({"town_refs", towns.empty() ? v_empty : v_items(V_TUPLE, -1, towns)});
    }
    const int32_t offers = record_of(T.offers, given);
    built.emplace(key, offers);
    return offers;
}

std::unordered_map<int32_t, std::unordered_map<int32_t, int32_t>> tile_places;  // tile tuple -> position value -> place

int32_t changed_tiles(int32_t previous, const std::vector<int32_t> &changes) {  // contract._changed_tiles
    if (changes.empty()) return previous;
    std::unordered_map<int32_t, int32_t> places;  // a copy: the table below is changed before this function ends
    {
        auto known = tile_places.find(previous);
        if (known != tile_places.end()) places = known->second;
        else for (size_t place = 0; place < V(previous).items.size(); place++)
            places.emplace(field(V(previous).items[place], T.tile_position), (int32_t)place);
    }
    std::vector<int32_t> items = V(previous).items;
    std::vector<bool> replaced(items.size(), false);
    for (int32_t tile : changes) {
        auto found = places.find(field(tile, T.tile_position));
        if (found == places.end() || replaced[(size_t)found->second] || items[(size_t)found->second] == tile) {
            PyErr_SetString(PyExc_ValueError, "A changed tile must replace a different tile of the previous view, once");
            throw Fail();
        }
        replaced[(size_t)found->second] = true;
        items[(size_t)found->second] = tile;
    }
    const int32_t made = v_items(V_TUPLE, -1, items);
    if (tile_places.size() > 64) tile_places.clear();
    tile_places[made] = std::move(places);  // a changed tile keeps its position
    return made;
}

struct Envelope {
    int64_t protocol = -1, player = -1;
    std::string type, episode, interface, history_schema, action_schema, army_information, message;
    bool has_army_information = false;
};

// One observation object: the fields of Observation, a missing map taken from the previous view, tile changes
// laid on the previous view's tiles (decode_observation with unchanged_map).
int32_t parse_observation(Json &reader, int32_t previous) {
    const VType &type = vtypes[(size_t)T.observation];
    std::vector<int32_t> items(type.fields.size(), -1);
    bool has_changes = false;
    std::vector<int32_t> changes;
    reader.expect('{');
    if (!reader.take('}')) {
        do {
            const std::string key = reader.string();
            reader.expect(':');
            if (key == "tile_changes") {
                has_changes = true;
                reader.expect('[');
                if (!reader.take(']')) { do changes.push_back(reader.record(T.tile)); while (reader.take(',')); reader.expect(']'); }
                continue;
            }
            auto found = type.by_name.find(key);
            if (found == type.by_name.end()) { PyErr_SetString(PyExc_ValueError, "Unknown fields in Observation"); reader.fail("unknown field"); }
            items[(size_t)found->second] = reader.value(type.fields[(size_t)found->second].spec);
        } while (reader.take(','));
        reader.expect('}');
    }
    const bool has_tiles = items[(size_t)T.o_tiles] >= 0, has_objects = items[(size_t)T.o_objects] >= 0;
    if (has_changes) {
        if (previous < 0 || has_tiles || !has_objects) {
            PyErr_SetString(PyExc_ValueError, "Tile changes need the previous view's tiles and this view's objects");
            throw Fail();
        }
        items[(size_t)T.o_tiles] = changed_tiles(field(previous, T.o_tiles), changes);
    } else if (!has_tiles && !has_objects && previous >= 0) {
        items[(size_t)T.o_tiles] = field(previous, T.o_tiles);
        items[(size_t)T.o_objects] = field(previous, T.o_objects);
    }
    for (size_t i = 0; i < items.size(); i++) {
        if (items[i] >= 0) continue;
        if (type.fields[i].fallback < 0) { PyErr_SetString(PyExc_ValueError, "Incomplete observation"); throw Fail(); }
        items[i] = type.fields[i].fallback;
    }
    return v_items(V_RECORD, T.observation, items);
}

// CGMine::flagMine sends this notice to the player taking ownership (protocol.Producer.ingest).
bool gold_mine_notice(int32_t event) {
    static int32_t notice_type = -1, n_mode, n_text, n_components, t_op, t_text_id, c_kind, c_resource;
    if (notice_type < 0) {
        notice_type = type_named("Notice");
        n_mode = field_of(notice_type, "display_mode"); n_text = field_of(notice_type, "text");
        n_components = field_of(notice_type, "components");
        const int32_t token = type_named("NoticeToken"), component = type_named("NoticeComponent");
        t_op = field_of(token, "op"); t_text_id = field_of(token, "text_id");
        c_kind = field_of(component, "kind"); c_resource = field_of(component, "resource");
    }
    if (!str_is(field(event, T.event_kind), "notice")) return false;
    const int32_t notice = field(event, T.event_notice);
    if (V(notice).kind != V_RECORD || !str_is(field(notice, n_mode), "auto")) return false;
    const std::vector<int32_t> &text = V(field(notice, n_text)).items;
    const std::vector<int32_t> &components = V(field(notice, n_components)).items;
    return text.size() == 1 && str_is(field(text[0], t_op), "append_text") && str_is(field(text[0], t_text_id), "core.mineevnt.6")
        && components.size() == 1 && str_is(field(components[0], c_kind), "resource_per_day")
        && str_is(field(components[0], c_resource), "gold");
}

// The raw text of an observation decodes to one value given the view before it when it names no map of its own,
// and to one value regardless of it otherwise: a text met again (about half the views repeat one) is not decoded again.
struct TextMemo { std::string text; int32_t previous; int32_t value; };
std::unordered_map<uint64_t, TextMemo> observation_texts;

int32_t observation_from_text(Json &reader, int32_t previous) {
    reader.ws();
    const char *start = reader.p;
    Json probe{reader.p, reader.end, {}};
    probe.skip();
    const size_t size = (size_t)(probe.p - start);
    uint64_t hash = 0x9E3779B97F4A7C15ULL ^ size;
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        uint64_t word;
        memcpy(&word, start + i, 8);
        hash = (hash ^ word) * 0xff51afd7ed558ccdULL;
        hash ^= hash >> 32;
    }
    for (; i < size; i++) hash = (hash ^ (uint8_t)start[i]) * 1099511628211ULL;
    const std::string_view text(start, size);
    const int32_t context = previous;  // a view without its own map takes the previous view's
    hash ^= (uint64_t)(uint32_t)context * 0x9E3779B97F4A7C15ULL;
    auto found = observation_texts.find(hash);
    if (found != observation_texts.end() && found->second.previous == context && found->second.text == text) {
        reader.p = probe.p;
        return found->second.value;
    }
    const int32_t value = parse_observation(reader, previous);
    if (observation_texts.size() > 65536) observation_texts.clear();
    observation_texts[hash] = TextMemo{std::string(text), context, value};
    return value;
}

void producer_failure(const char *text) { PyErr_SetString(PyExc_ValueError, text); throw Fail(); }

PyObject *native_failure = nullptr;  // protocol.NativeFailure

void ingest_record(Producer *self, Json &reader) {
    NFrame frame;
    int64_t native_id = -1;
    bool has_id = false, has_observation = false;
    int kind = -1;
    reader.expect('{');
    // One pass: nothing read here depends on the kind (the decision id is mapped after the record).
    if (!reader.take('}')) {
        do {
            const std::string key = reader.string();
            reader.expect(':');
            if (key == "kind") {
                const std::string &name = reader.string();
                for (int k = 0; k < F_SUBMITTED; k++) if (name == KIND_NAMES[k]) kind = k;
                if (kind < 0) producer_failure("Unsupported contextual record kind");
            }
            else if (key == "day") frame.day = (int32_t)reader.integer();
            else if (key == "decision_id") { native_id = reader.integer(); has_id = true; }
            else if (key == "revision") { frame.revision = reader.integer(); frame.has_revision = true; }
            else if (key == "command_index") { frame.command_index = reader.integer(); frame.has_command = true; }
            else if (key == "accepted") { if (reader.literal("true")) frame.accepted = 1; else if (reader.literal("false")) frame.accepted = 0; else producer_failure("accepted"); }
            else if (key == "consumer") { const std::string &text = reader.string(); frame.consumer = string_id(text.data(), text.size()); }
            else if (key == "reason") { const std::string &text = reader.string(); frame.reason = string_id(text.data(), text.size()); }
            else if (key == "outcome") { const std::string &text = reader.string(); frame.outcome = string_id(text.data(), text.size()); }
            else if (key == "observation") { frame.observation = observation_from_text(reader, self->view); has_observation = true; }
            else if (key == "context") frame.context = reader.record(T.context);
            else if (key == "result") frame.result = reader.record(T.result);
            else if (key == "event") frame.event = reader.record(T.event);
            else producer_failure("Unexpected or missing record fields");
        } while (reader.take(','));
        reader.expect('}');
    }
    if (kind < 0) producer_failure("Unsupported contextual record kind");
    frame.kind = (FKind)kind;
    frame.seq = (int32_t)self->frames->size();
    if (has_id) {
        if (native_id < 0) producer_failure("decision_id must be a nonnegative native-local integer");
        if (frame.kind == F_DECISION) {
            if (self->decision_ids->count(native_id)) producer_failure("Repeated native decision_id");
        } else {
            auto found = self->decision_ids->find(native_id);
            if (found == self->decision_ids->end()) producer_failure("Unknown native decision reference");
            frame.decision_seq = found->second;
        }
    }
    if (frame.kind == F_DECISION) {
        if (!has_observation) {
            if (self->view < 0) producer_failure("A decision without an observation needs an earlier view");
            frame.observation = self->view;
        }
        frame.offers = offers_of(frame.observation, *self->action_schema);
    }
    self->frames->push_back(frame);
    if (frame.kind == F_VIEW) self->view = frame.observation;
    else if (frame.kind == F_DECISION) { (*self->decision_ids)[native_id] = frame.seq; self->current_id = native_id; self->has_current = true; }
    else if (frame.kind == F_RESULT || frame.kind == F_SUPERSEDED) self->has_current = false;
    else if (frame.kind == F_TERMINAL) self->terminal = frame.outcome;
    else if (frame.kind == F_EVENT && gold_mine_notice(frame.event)) self->gold_mine_captures++;
}

void ingest_message(Producer *self, const char *bytes, Py_ssize_t size) {
    prepare_types();
    Json reader{bytes, bytes + size, {}};
    Envelope envelope;
    const char *records_start = nullptr, *records_end = nullptr;
    reader.expect('{');
    if (!reader.take('}')) {
        do {
            const std::string key = reader.string();
            reader.expect(':');
            if (key == "protocol_version") envelope.protocol = reader.integer();
            else if (key == "type") envelope.type = reader.string();
            else if (key == "episode") envelope.episode = reader.string();
            else if (key == "player") envelope.player = reader.integer();
            else if (key == "interface") envelope.interface = reader.string();
            else if (key == "history_schema") envelope.history_schema = reader.string();
            else if (key == "action_schema") envelope.action_schema = reader.string();
            else if (key == "army_information") { envelope.army_information = reader.string(); envelope.has_army_information = true; }
            else if (key == "message") envelope.message = reader.string();
            else if (key == "records") { reader.ws(); records_start = reader.p; reader.skip(); records_end = reader.p; }
            else producer_failure("Unexpected envelope fields");
        } while (reader.take(','));
        reader.expect('}');
    }
    if (envelope.protocol != self->protocol) producer_failure("Contextual mode requires its explicit protocol_version");
    if (envelope.episode != *self->episode || envelope.player != self->player) producer_failure("Cross-player or cross-episode contextual envelope");
    if (envelope.type == "ready") {
        if (self->ready || envelope.interface != *self->interface || envelope.history_schema != *self->history_schema
                || envelope.action_schema != *self->action_schema || envelope.has_army_information != !self->army_information->empty()
                || envelope.army_information != *self->army_information)
            producer_failure("Duplicate ready or mismatched contextual schema");
        self->ready = true;
        return;
    }
    if (envelope.type == "error") {
        PyErr_SetString(native_failure ? native_failure : PyExc_ValueError, envelope.message.c_str());
        throw Fail();
    }
    if (envelope.type != "contextual" || !self->ready || !records_start) producer_failure("Expected a nonempty contextual record batch after ready");
    Json records{records_start, records_end, {}};
    records.expect('[');
    if (records.take(']')) producer_failure("Expected a nonempty contextual record batch after ready");
    do ingest_record(self, records); while (records.take(','));
    records.expect(']');
}

int Producer_init(Producer *self, PyObject *args, PyObject *) {
    const char *episode, *interface, *history_schema, *action_schema, *army_information;
    long long player, protocol;
    if (!PyArg_ParseTuple(args, "sLLssss", &episode, &player, &protocol, &interface, &history_schema, &action_schema,
                          &army_information))
        return -1;
    self->frames = new std::vector<NFrame>();
    self->decision_ids = new std::unordered_map<int64_t, int32_t>();
    self->episode = new std::string(episode);
    self->interface = new std::string(interface);
    self->history_schema = new std::string(history_schema);
    self->action_schema = new std::string(action_schema);
    self->army_information = new std::string(army_information);
    self->player = player;
    self->protocol = protocol;
    self->python_frames = PyList_New(0);
    return self->python_frames ? 0 : -1;
}

void Producer_dealloc(Producer *self) {
    delete self->frames; delete self->decision_ids; delete self->episode; delete self->interface;
    delete self->history_schema; delete self->action_schema; delete self->army_information;
    Py_XDECREF(self->python_frames);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *Producer_ingest(Producer *self, PyObject *data) {
    Py_buffer buffer;
    if (PyObject_GetBuffer(data, &buffer, PyBUF_SIMPLE)) return nullptr;
    const size_t before = self->frames->size();
    try {
        ingest_message(self, (const char *)buffer.buf, buffer.len);
    } catch (JsonError &) {
        PyBuffer_Release(&buffer);
        return nullptr;
    } catch (Fail &) {
        PyBuffer_Release(&buffer);
        return nullptr;
    }
    PyBuffer_Release(&buffer);
    return PyLong_FromSize_t(self->frames->size() - before);
}

PyObject *python_frame(Producer *self, const NFrame &frame) {
    static int32_t frame_type = -1;
    if (frame_type < 0) frame_type = type_named("Frame");
    Ref keywords(PyDict_New());
    auto put = [&](const char *name, PyObject *value) {  // value borrowed
        if (PyDict_SetItemString(keywords.get(), name, value)) throw Fail();
    };
    auto put_new = [&](const char *name, PyObject *value) { Ref held(value); put(name, held.get()); };
    put_new("episode", PyUnicode_FromString(self->episode->c_str()));
    put_new("player", PyLong_FromLongLong(self->player));
    put_new("seq", PyLong_FromLong(frame.seq));
    put_new("day", PyLong_FromLong(frame.day));
    put_new("kind", PyUnicode_FromString(KIND_NAMES[frame.kind]));
    put_new("schema", PyUnicode_FromString(self->history_schema->c_str()));
    if (frame.observation >= 0) put("observation", python_value(frame.observation));
    if (frame.context >= 0) put("context", python_value(frame.context));
    if (frame.offers >= 0) put("offers", python_value(frame.offers));
    if (frame.action >= 0) put("action", python_value(frame.action));
    if (frame.result >= 0) put("result", python_value(frame.result));
    if (frame.event >= 0) put("event", python_value(frame.event));
    if (frame.decision_seq >= 0) put_new("decision_seq", PyLong_FromLong(frame.decision_seq));
    if (frame.has_revision) put_new("revision", PyLong_FromLongLong(frame.revision));
    if (frame.has_command) put_new("command_index", PyLong_FromLongLong(frame.command_index));
    if (frame.consumer >= 0) put("consumer", string_object(frame.consumer));
    if (frame.reason >= 0) put("reason", string_object(frame.reason));
    if (frame.outcome >= 0) put("outcome", string_object(frame.outcome));
    if (frame.accepted >= 0) put("accepted", frame.accepted ? Py_True : Py_False);
    Ref arguments(PyTuple_New(0));
    PyObject *made = PyObject_Call(vtypes[(size_t)frame_type].python, arguments.get(), keywords.get());
    if (!made) throw Fail();
    return made;
}

// python_frames(): the contract Frames of every frame so far (converted once, in order)
PyObject *Producer_python_frames(Producer *self, PyObject *) {
    try {
        for (size_t i = (size_t)PyList_GET_SIZE(self->python_frames); i < self->frames->size(); i++) {
            Ref made(python_frame(self, (*self->frames)[i]));
            if (PyList_Append(self->python_frames, made.get())) throw Fail();
        }
    } catch (Fail &) {
        return nullptr;
    }
    Py_INCREF(self->python_frames);
    return self->python_frames;
}

// submitted(action, day): the frame of the answer written to the engine, for the decision held now
PyObject *Producer_submitted(Producer *self, PyObject *args) {
    PyObject *action;
    long day;
    if (!PyArg_ParseTuple(args, "Ol", &action, &day)) return nullptr;
    try {
        prepare_types();
        int32_t decision = -1;
        for (size_t i = self->frames->size(); i-- > 0;) if ((*self->frames)[i].kind == F_DECISION) { decision = (int32_t)i; break; }
        if (decision < 0) producer_failure("Action does not match the current contextual request");
        NFrame frame;
        frame.kind = F_SUBMITTED;
        frame.seq = (int32_t)self->frames->size();
        frame.day = self->frames->empty() ? (int32_t)day : self->frames->back().day;  // PlayerHistory.emit: the history's day
        frame.decision_seq = decision;
        frame.action = value_of_python(action, [&] {
            for (size_t i = 0; i < specs.size(); i++) if (specs[i].kind == S_RECORD && specs[i].type == T.action) return (int32_t)i;
            producer_failure("no Action spec");
            return -1;
        }());
        self->frames->push_back(frame);
    } catch (Fail &) {
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject *Producer_get(Producer *self, void *which) {
    switch ((intptr_t)which) {
        case 0: return PyLong_FromSize_t(self->frames->size());
        case 1: if (self->terminal < 0) Py_RETURN_NONE; try { PyObject *t = string_object(self->terminal); Py_INCREF(t); return t; } catch (Fail &) { return nullptr; }
        case 2: return PyBool_FromLong(self->ready);
        case 3: return PyLong_FromLong(self->gold_mine_captures);
        case 4: if (!self->has_current) Py_RETURN_NONE; return PyLong_FromLongLong(self->current_id);
    }
    Py_RETURN_NONE;
}

PyObject *Producer_observation_ids(Producer *self, PyObject *) {  // tests: each frame's observation value, -1 without
    PyObject *out = PyList_New((Py_ssize_t)self->frames->size());
    if (!out) return nullptr;
    for (size_t i = 0; i < self->frames->size(); i++) PyList_SET_ITEM(out, (Py_ssize_t)i, PyLong_FromLong((*self->frames)[i].observation));
    return out;
}

PyGetSetDef Producer_getset[] = {
    {"frame_count", (getter)Producer_get, nullptr, nullptr, (void *)0},
    {"terminal", (getter)Producer_get, nullptr, nullptr, (void *)1},
    {"ready", (getter)Producer_get, nullptr, nullptr, (void *)2},
    {"gold_mine_captures", (getter)Producer_get, nullptr, nullptr, (void *)3},
    {"current_id", (getter)Producer_get, nullptr, nullptr, (void *)4},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyMethodDef Producer_methods[] = {
    {"ingest", (PyCFunction)Producer_ingest, METH_O, "ingest(bytes): fold one engine message; the number of new frames"},
    {"python_frames", (PyCFunction)Producer_python_frames, METH_NOARGS, "the contract Frames so far"},
    {"submitted", (PyCFunction)Producer_submitted, METH_VARARGS, "submitted(action, day): the frame of an answer sent"},
    {"observation_ids", (PyCFunction)Producer_observation_ids, METH_NOARGS, "each frame's observation value (tests)"},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject ProducerType = {PyVarObject_HEAD_INIT(nullptr, 0)};

PyObject *py_set_native_failure(PyObject *, PyObject *type) {
    Py_XDECREF(native_failure);
    native_failure = type;
    Py_INCREF(type);
    Py_RETURN_NONE;
}

// == semantic trees of values ======================================================================================
// policy._semantic of the contract object a value stands for, memoized by value: a value's tree is built once.

std::vector<int32_t> tree_of_value;  // value -> tree, -1 until built
std::vector<int8_t> plan_ready;
std::vector<Plan> value_plans;       // by type index
struct SpecialTypes { int32_t recruitment = -1, tavern = -1, stack = -1, dialog = -1, dialog_choice = -1, notice = -1; bool ready = false; } ST;
struct PanelFields {
    int32_t offers, selected_offer_ref, selected_upgrade_ref, stock, unit_cost, slider_max, destination, selected_detail,
        gold_cost, recruit_status, parent_context, source, total_cost, affordable, family, text, components, selection,
        cancel, choices, display_mode, ref_recruit, ref_tavern, ref_upgrade;
} PF;

int32_t vsemantic(int32_t value);

Plan &value_plan(int32_t type) {
    if ((size_t)type >= value_plans.size()) { value_plans.resize(vtypes.size()); plan_ready.resize(vtypes.size(), 0); }
    if (!plan_ready[(size_t)type]) {
        Plan &plan = plan_of((PyTypeObject *)vtypes[(size_t)type].python);  // policy._field_plan of the dataclass
        value_plans[(size_t)type] = plan;
        // each field's index in the value
        plan_ready[(size_t)type] = 1;
    }
    return value_plans[(size_t)type];
}

int32_t field_index(int32_t type, PyObject *name) {
    Py_ssize_t size;
    const char *chars = PyUnicode_AsUTF8AndSize(name, &size);
    return vtypes[(size_t)type].by_name.at(std::string(chars, (size_t)size));
}

void prepare_special() {
    if (ST.ready) return;
    ST.recruitment = type_named("RecruitmentPanel"); ST.tavern = type_named("TavernPanel"); ST.stack = type_named("StackPanel");
    ST.dialog = type_named("DialogPanel"); ST.dialog_choice = type_named("DialogChoice"); ST.notice = type_named("Notice");
    ST.ready = true;
}

int32_t vtext(int32_t value) {  // _text of a str value
    const std::string &text = strings[(size_t)V(value).string];
    return text_node(text.data(), (Py_ssize_t)text.size());
}

int32_t vnumber(int32_t value) {  // _number of an int value: float(int)
    return number_node((double)V(value).number);
}

int32_t vfield(int32_t value, int32_t type, const char *name) { return field(value, field_of(type, name)); }

int32_t vcards(int32_t panel, int32_t type, const char *selected_name, const char *card_tag, const char *set_name) {
    const int32_t selected = vfield(panel, type, selected_name);
    const std::vector<int32_t> offers = V(vfield(panel, type, "offers")).items;
    std::vector<int32_t> out;
    const int32_t tag = text_node(card_tag);
    for (int32_t card : offers) {
        const int32_t card_type = V(card).type;
        const int32_t ref = vfield(card, card_type, "ref");
        out.push_back(node(tag, {vsemantic(card), truth(ref == selected)}));
    }
    return sorted_set(set_name, out);
}

int32_t vdialog_choice(int32_t panel, int32_t choice) {
    const int32_t choice_type = V(choice).type;
    const int32_t kind = vfield(choice, choice_type, "kind");
    int32_t component;
    if (str_is(kind, "select_component")) {
        const int64_t index = V(vfield(choice, choice_type, "component_index")).number;
        const std::vector<int32_t> &components = V(vfield(panel, ST.dialog, "components")).items;
        if (index < 0 || (size_t)index >= components.size()) { PyErr_SetString(PyExc_IndexError, "tuple index out of range"); throw Fail(); }
        component = vsemantic(components[(size_t)index]);
    } else {
        component = node("absent");
    }
    return node("dialog_choice", {vtext(kind), component});
}

int32_t vspecial(int32_t value) {
    prepare_special();
    const int32_t type = V(value).type;
    if (type == ST.recruitment) {
        const int32_t set = vcards(value, type, "selected_offer_ref", "card", "cards");
        return node("recruitment", {set, node("stock", {vnumber(vfield(value, type, "stock"))}),
                                    node("unit_cost", {vsemantic(vfield(value, type, "unit_cost"))}),
                                    node("slider_max", {vnumber(vfield(value, type, "slider_max"))}),
                                    vsemantic(vfield(value, type, "destination"))});
    }
    if (type == ST.tavern) {
        const int32_t set = vcards(value, type, "selected_offer_ref", "card", "cards");
        return node("tavern", {set, node("selected_detail", {vsemantic(vfield(value, type, "selected_detail"))}),
                               node("gold_cost", {vnumber(vfield(value, type, "gold_cost"))}),
                               node("recruit_status", {vtext(vfield(value, type, "recruit_status"))})});
    }
    if (type == ST.stack) {
        const int32_t first = vsemantic(vfield(value, type, "parent_context")), second = vsemantic(vfield(value, type, "source"));
        const int32_t set = vcards(value, type, "selected_upgrade_ref", "upgrade", "offers");
        return node("stack_panel", {first, second, set, node("total_cost", {vsemantic(vfield(value, type, "total_cost"))}),
                                    node("affordable", {vsemantic(vfield(value, type, "affordable"))})});
    }
    if (type == ST.dialog) {
        const int32_t family = vtext(vfield(value, type, "family")), text = vsemantic(vfield(value, type, "text"));
        std::vector<int32_t> components;
        for (int32_t item : V(vfield(value, type, "components")).items) components.push_back(vsemantic(item));
        const int32_t component_set = sorted_set("components", components);
        const int32_t selection = node("selection", {vsemantic(vfield(value, type, "selection"))});
        const int32_t cancel = node("cancel", {vsemantic(vfield(value, type, "cancel"))});
        std::vector<int32_t> choices;
        for (int32_t item : V(vfield(value, type, "choices")).items) choices.push_back(vdialog_choice(value, item));
        return node("dialog_panel", {family, text, component_set, selection, cancel, sorted_set("choices", choices)});
    }
    if (type == ST.dialog_choice) { PyErr_SetString(PyExc_ValueError, "Dialog choice requires its displayed component binding"); throw Fail(); }
    if (type == ST.notice) {
        const int32_t mode = vtext(vfield(value, type, "display_mode"));
        std::vector<int32_t> tokens, components;
        for (int32_t item : V(vfield(value, type, "text")).items) tokens.push_back(vsemantic(item));
        const int32_t ordered = interior_node(ORDERED, text_node("notice_text"), tokens);
        for (int32_t item : V(vfield(value, type, "components")).items) components.push_back(vsemantic(item));
        return node("notice", {mode, ordered, sorted_set("components", components)});
    }
    return -1;
}

int32_t category_of(int64_t number) {
    const std::string text = "category:" + std::to_string(number);
    return text_node(text.data(), (Py_ssize_t)text.size());
}

int32_t vsemantic(int32_t value) {
    if ((size_t)value < tree_of_value.size() && tree_of_value[(size_t)value] >= 0) return tree_of_value[(size_t)value];
    const Val &item = V(value);
    int32_t tree;
    switch (item.kind) {
        case V_NONE: tree = node("absent"); break;
        case V_BOOL: tree = truth(item.number != 0); break;
        case V_INT: tree = vnumber(value); break;
        case V_STR: tree = vtext(value); break;
        case V_TUPLE: {
            const std::vector<int32_t> items = item.items;
            std::vector<int32_t> children;
            for (int32_t entry : items) children.push_back(vsemantic(entry));
            tree = interior_node(ORDERED, text_node("tuple"), children);
            break;
        }
        default: {
            tree = vspecial(value);
            if (tree >= 0) break;
            const int32_t type = V(value).type;
            Plan &plan = value_plan(type);
            std::vector<int32_t> children;
            for (const Plan::Field &entry : plan.fields) {
                const int32_t raw = field(value, field_index(type, entry.name));
                const Val &raw_value = V(raw);
                if (entry.absent_skipped && raw_value.kind == V_NONE) continue;
                int32_t encoded;
                if (entry.category && raw_value.kind == V_INT) {
                    encoded = category_of(raw_value.number);
                } else if (entry.amount && raw_value.kind == V_INT) {
                    encoded = str_is(vfield(value, type, "kind"), "world_object") ? category_of(raw_value.number) : vsemantic(raw);
                } else if (entry.as_set && raw_value.kind == V_TUPLE) {
                    const std::vector<int32_t> items = raw_value.items;
                    std::vector<int32_t> members;
                    for (int32_t member : items) members.push_back(vsemantic(member));
                    encoded = sorted_set(entry.tag, members);
                } else {
                    encoded = vsemantic(raw);
                }
                children.push_back(node(entry.tag, {encoded}));
            }
            tree = node(plan.type_tag, children);
        }
    }
    if ((size_t)value >= tree_of_value.size()) tree_of_value.resize(vals.size() + 4096, -1);
    tree_of_value[(size_t)value] = tree;
    return tree;
}

// observation trees of values (policy._observation_semantics)
std::unordered_map<int32_t, std::unordered_map<int32_t, int32_t>> handle_tables;  // observation -> handle string -> entry

int32_t vsubject(int32_t observation, int32_t ref) {
    if (observation < 0 || V(ref).kind == V_NONE) return node("unresolved_reference");
    auto &table = handle_tables[observation];
    if (table.empty()) {
        for (int32_t group : {field(observation, T.o_heroes), field(observation, T.o_towns), field(observation, T.o_objects)})
            for (int32_t entry : V(group).items)
                table.emplace(V(vfield(entry, V(entry).type, "handle")).string, entry);
        table.emplace(-1, -1);  // marks the table as built (no handle is string -1)
    }
    if (handle_tables.size() > 4096) { std::unordered_map<int32_t, int32_t> keep = table; handle_tables.clear(); handle_tables[observation] = keep; }
    auto &built = handle_tables[observation];
    auto found = built.find(V(ref).string);
    if (found == built.end() || V(ref).kind != V_STR) return node("unresolved_reference");
    return vsemantic(found->second);
}

std::unordered_map<int32_t, int32_t> observation_trees;

int32_t vobservation_tree(int32_t observation) {
    auto known = observation_trees.find(observation);
    if (known != observation_trees.end()) return known->second;
    const int32_t window = field(observation, T.o_window);
    const bool windowed = V(window).kind == V_RECORD;
    std::vector<int32_t> heroes, towns;
    for (int32_t hero : V(field(observation, T.o_heroes)).items) heroes.push_back(vsemantic(hero));
    for (int32_t town : V(field(observation, T.o_towns)).items) towns.push_back(vsemantic(town));
    const int32_t size_node = node("size", {vsemantic(field(observation, T.o_size))});
    const int32_t resource_node = node("resources", {vsemantic(vfield(observation, T.observation, "resources"))});
    const int32_t hero_set = sorted_set("heroes", heroes), town_set = sorted_set("towns", towns);
    int32_t window_node = vsemantic(window);
    if (windowed) {
        const int32_t panel = field(window, T.w_capability_panel);
        if (V(panel).kind == V_RECORD) {
            const int32_t panel_type = V(panel).type;
            std::vector<int32_t> owned, bound;
            const int32_t owned_tag = text_node("owned_item");
            for (int32_t item : V(vfield(panel, panel_type, "items")).items)
                owned.push_back(node(owned_tag, {vsemantic(item), vsubject(observation, vfield(item, V(item).type, "subject"))}));
            for (int32_t choice : V(vfield(panel, panel_type, "choices")).items) {
                const int32_t choice_type = V(choice).type;
                const int32_t destination = vfield(choice, choice_type, "destination");
                bound.push_back(node("capability_choice", {vsemantic(choice), node("actor", {vsubject(observation, vfield(choice, choice_type, "subject"))}),
                                                           V(destination).kind != V_NONE ? node("recipient", {vsubject(observation, destination)}) : node("no_recipient")}));
            }
            window_node = node("capability_window", {window_node, sorted_set("bound_items", owned), sorted_set("bound_choices", bound),
                                                     node("inspected_object", {vsubject(observation, vfield(panel, panel_type, "inspected_subject"))})});
        }
    }
    const int32_t context = windowed ? field(window, T.w_context) : default_context();
    const int32_t subject_node = node("current_window_subject", {vsubject(observation, vfield(context, T.context, "subject"))});
    int32_t own = node("observation", {size_node, resource_node, hero_set, town_set, window_node, subject_node});
    if (windowed && V(field(window, T.w_armies)).kind != V_NONE) {
        std::vector<int32_t> bindings;
        const int32_t tag = text_node("displayed_army");
        for (int32_t army : V(field(window, T.w_armies)).items)
            bindings.push_back(node(tag, {vsemantic(army), vsubject(observation, vfield(army, V(army).type, "subject"))}));
        const int32_t stack = field(window, T.w_stack_panel);
        if (V(stack).kind == V_RECORD) {
            const int32_t parent = vfield(stack, V(stack).type, "parent_context");
            const int32_t first = vsubject(observation, vfield(parent, T.context, "subject"));
            const int32_t second = str_is(field(parent, T.c_kind), "exchange") ? vsubject(observation, vfield(parent, T.context, "selection")) : node("absent");
            bindings.push_back(node("stack_parent", {first, second}));
        }
        if (str_is(field(context, T.c_kind), "exchange"))
            bindings.push_back(node("exchange_partner", {vsubject(observation, vfield(context, T.context, "selection"))}));
        own = node("management_observation", {own, sorted_set("army_bindings", bindings)});
    }
    const int32_t armies = field(observation, T.o_own_armies);
    if (V(armies).kind != V_NONE) {
        std::vector<int32_t> bindings;
        const int32_t tag = text_node("own_army");
        for (int32_t army : V(armies).items)
            bindings.push_back(node(tag, {vsemantic(army), vsubject(observation, vfield(army, V(army).type, "subject"))}));
        own = node("passive_army_observation", {own, sorted_set("own_army_bindings", bindings)});
    }
    if (observation_trees.size() > 65536) observation_trees.clear();
    observation_trees[observation] = own;
    return own;
}

PyObject *py_value_tree(PyObject *, PyObject *args) {  // value_tree(value id, observation?): tests
    long value, observation_flag = 0;
    if (!PyArg_ParseTuple(args, "l|l", &value, &observation_flag)) return nullptr;
    try {
        prepare_types();
        PyObject *tree = python_tree(observation_flag ? vobservation_tree((int32_t)value) : vsemantic((int32_t)value));
        Py_INCREF(tree);
        return tree;
    } catch (Fail &) {
        return nullptr;
    }
}

// == the seat's encoder: payload trees of frames (policy.ContextualEncoder.encode) ================================

int32_t first_slot(int32_t armies, int32_t ref, int32_t *army_out) {  // contract.army_slot: the first army and slot
    for (int32_t army : V(armies).items) {
        const int32_t army_type = V(army).type;
        for (int32_t slot : V(vfield(army, army_type, "slots")).items)
            if (vfield(slot, V(slot).type, "ref") == ref) { *army_out = army; return slot; }
    }
    *army_out = -1;
    return -1;
}

int32_t varmy_slot(int32_t observation, int32_t ref, bool receiving) {  // policy._describe_army_slot
    const int32_t window = observation >= 0 ? field(observation, T.o_window) : v_none;
    int32_t army = -1, slot = -1;
    if (V(window).kind == V_RECORD && V(field(window, T.w_armies)).kind != V_NONE) {
        slot = first_slot(field(window, T.w_armies), ref, &army);
    } else {
        const int32_t panel = V(window).kind == V_RECORD ? field(window, T.w_town_panel) : v_none;
        const int32_t exchange = V(panel).kind == V_RECORD ? vfield(panel, V(panel).type, "armies") : v_none;
        if (V(exchange).kind == V_RECORD) {
            army = vfield(exchange, V(exchange).type, receiving ? "destination" : "source");
            if (V(army).kind == V_RECORD)
                for (int32_t item : V(vfield(army, V(army).type, "slots")).items)
                    if (vfield(item, V(item).type, "ref") == ref) { slot = item; break; }
        }
    }
    if (slot < 0) { PyErr_SetString(PyExc_ValueError, "Army slot has no current displayed semantic binding"); throw Fail(); }
    const int32_t army_type = V(army).type;
    return node("army_slot", {node("role", {vtext(vfield(army, army_type, "role"))}),
                              node("owner_subject", {vsubject(observation, vfield(army, army_type, "subject"))}),
                              node("requires_nonempty", {vsemantic(vfield(army, army_type, "requires_nonempty"))}), vsemantic(slot)});
}

int32_t vselection(int32_t observation, int32_t ref) {  // policy._selection
    const int32_t unresolved = node("unresolved_selection");
    if (observation < 0) return unresolved;
    const int32_t window = field(observation, T.o_window);
    if (V(window).kind != V_RECORD) return unresolved;
    const int32_t capability = field(window, T.w_capability_panel);
    if (V(capability).kind == V_RECORD) {
        for (int32_t choice : V(vfield(capability, V(capability).type, "choices")).items) {
            const int32_t choice_type = V(choice).type;
            if (vfield(choice, choice_type, "ref") != ref) continue;
            const int32_t destination = vfield(choice, choice_type, "destination");
            return node("capability_choice", {vsemantic(choice), node("actor", {vsubject(observation, vfield(choice, choice_type, "subject"))}),
                                              V(destination).kind != V_NONE ? node("recipient", {vsubject(observation, destination)}) : node("no_recipient")});
        }
        return unresolved;
    }
    const int32_t dialog = field(window, T.w_dialog_panel);
    if (V(dialog).kind == V_RECORD) {
        for (int32_t choice : V(vfield(dialog, V(dialog).type, "choices")).items)
            if (vfield(choice, V(choice).type, "ref") == ref) return vdialog_choice(dialog, choice);
        return unresolved;
    }
    const int32_t stack = field(window, T.w_stack_panel);
    if (V(stack).kind == V_RECORD) {
        for (int32_t offer : V(vfield(stack, V(stack).type, "offers")).items)
            if (vfield(offer, V(offer).type, "ref") == ref) return vsemantic(offer);
        return unresolved;
    }
    const int32_t panel = field(window, T.w_town_panel);
    if (V(panel).kind != V_RECORD) return unresolved;
    const int32_t panel_type = V(panel).type;
    std::vector<int32_t> entries;
    for (int32_t item : V(vfield(panel, panel_type, "controls")).items) entries.push_back(item);
    for (int32_t item : V(vfield(panel, panel_type, "buildings")).items) entries.push_back(item);
    const int32_t building = vfield(panel, panel_type, "building");
    if (V(building).kind == V_RECORD) entries.push_back(building);
    const int32_t recruitment = vfield(panel, panel_type, "recruitment");
    if (V(recruitment).kind == V_RECORD) for (int32_t item : V(vfield(recruitment, V(recruitment).type, "offers")).items) entries.push_back(item);
    const int32_t tavern = vfield(panel, panel_type, "tavern");
    if (V(tavern).kind == V_RECORD) for (int32_t item : V(vfield(tavern, V(tavern).type, "offers")).items) entries.push_back(item);
    for (int32_t entry : entries) if (vfield(entry, V(entry).type, "ref") == ref) return vsemantic(entry);
    return unresolved;
}

bool kind_in(int32_t kind, std::initializer_list<const char *> names) {
    for (const char *name : names) if (str_is(kind, name)) return true;
    return false;
}

int32_t vaction(int32_t action, int32_t observation) {  // policy._action
    const int32_t kind = field(action, field_of(T.action, "kind"));
    std::vector<int32_t> children = {vtext(kind), node("decision_subject", {vsubject(observation, vfield(action, T.action, "subject"))})};
    for (const char *key : {"x", "y", "count"}) children.push_back(node(key, {vsemantic(vfield(action, T.action, key))}));
    children.push_back(node("panel", {vsemantic(vfield(action, T.action, "panel"))}));
    const int32_t selection = vfield(action, T.action, "selection");
    const bool slots = kind_in(kind, {"transfer_stack", "split_stack", "merge_stacks", "swap_stacks", "open_stack"});
    children.push_back(node("decision_selection", {slots ? varmy_slot(observation, selection, false) : vselection(observation, selection)}));
    const int32_t destination = vfield(action, T.action, "destination");
    if (V(destination).kind != V_NONE) {
        if (kind_in(kind, {"transfer_stack", "split_stack", "merge_stacks", "swap_stacks"})) {
            children.push_back(node("destination", {varmy_slot(observation, destination, true)}));
        } else {
            const int32_t panel = field(field(observation, T.o_window), T.w_town_panel);
            const int32_t recruitment = vfield(panel, V(panel).type, "recruitment");
            const int32_t target = V(recruitment).kind == V_RECORD ? vfield(recruitment, V(recruitment).type, "destination") : -1;
            if (target < 0 || vfield(target, V(target).type, "ref") != destination) {
                PyErr_SetString(PyExc_ValueError, "Action destination has no displayed semantic binding");
                throw Fail();
            }
            children.push_back(node("destination", {vsemantic(target)}));
        }
    }
    return node("action", children);
}

// the payload of a frame, -1 for the bookkeeping kinds the model skips (policy._METADATA)
int32_t vpayload(const NFrame &frame, int32_t basis, int32_t observation) {
    if (frame.kind == F_ISSUED || frame.kind == F_SUPERSEDED) return -1;
    std::vector<int32_t> children = {node("day", {number_node((double)frame.day)}), text_node(KIND_NAMES[frame.kind])};
    if (frame.action >= 0) children.push_back(vaction(frame.action, basis));
    if (frame.event >= 0) {
        children.push_back(vsemantic(frame.event));
        const int32_t subject = vfield(frame.event, T.event, "subject"), partner = vfield(frame.event, T.event, "partner");
        if (V(subject).kind != V_NONE) children.push_back(node("prior_observation_subject", {vsubject(observation, subject)}));
        if (V(partner).kind != V_NONE) children.push_back(node("prior_observation_partner", {vsubject(observation, partner)}));
    }
    if (frame.result >= 0) children.push_back(vsemantic(frame.result));
    if (frame.context >= 0) {
        children.push_back(vsemantic(frame.context));
        children.push_back(node("prior_observation_subject", {vsubject(observation, vfield(frame.context, T.context, "subject"))}));
        if (str_is(field(frame.context, T.c_kind), "exchange"))
            children.push_back(node("prior_observation_partner", {vsubject(observation, vfield(frame.context, T.context, "selection"))}));
    }
    if (frame.accepted >= 0) children.push_back(node("accepted", {truth(frame.accepted == 1)}));
    if (frame.outcome >= 0) children.push_back(node("observed_outcome", {text_node(strings[(size_t)frame.outcome].data(), (Py_ssize_t)strings[(size_t)frame.outcome].size())}));
    return node("frame", children);
}

PyObject *py_payloads(PyObject *, PyObject *object) {  // payloads(producer): each frame's payload tree (tests)
    if (Py_TYPE(object) != &ProducerType) { PyErr_SetString(PyExc_TypeError, "a Producer"); return nullptr; }
    Producer *producer = (Producer *)object;
    try {
        prepare_types();
        Ref out(PyList_New(0));
        int32_t decision_observation = -1, observation = -1;
        bool held = false;
        for (const NFrame &frame : *producer->frames) {
            const int32_t basis = held ? decision_observation : observation;
            const int32_t payload = vpayload(frame, basis, observation);
            if (frame.kind == F_DECISION) { held = true; decision_observation = frame.observation; }
            else if (frame.kind == F_RESULT || frame.kind == F_SUPERSEDED) held = false;
            if (frame.observation >= 0) observation = frame.observation;
            if (PyList_Append(out.get(), payload >= 0 ? python_tree(payload) : Py_None)) throw Fail();
        }
        Py_INCREF(out.get());
        return out.get();
    } catch (Fail &) {
        return nullptr;
    }
}

std::unordered_map<uint64_t, int32_t> field_cache;

inline int32_t F(int32_t value, const char *name) {  // value.name, for a literal name
    const Val &item = V(value);
    if (item.kind != V_RECORD) { PyErr_Format(PyExc_AttributeError, "no attribute '%s' on a %s", name, item.kind == V_NONE ? "None" : "non-record"); throw Fail(); }
    const uint64_t key = ((uint64_t)item.type << 48) ^ (uint64_t)(uintptr_t)name;
    auto found = field_cache.find(key);
    int32_t index;
    if (found == field_cache.end()) { index = field_of(item.type, name); field_cache.emplace(key, index); }
    else index = found->second;
    return item.items[(size_t)index];
}
inline const std::vector<int32_t> &items_of(int32_t value) { return V(value).items; }

// == the interaction guard (heroes.game.guard) =====================================================================
// Keys of strings are joined with \x1f, below every character the engine sends, so joined keys compare as the
// tuples do; a None element is \x01NONE\x01.

typedef std::string Key;
const char SEP = '\x1f';
const std::string NONE_ELEMENT = "\x01NONE\x01";

std::string text_or_none(int32_t value) {
    const Val &item = V(value);
    if (item.kind == V_NONE) return NONE_ELEMENT;
    if (item.kind == V_STR) return strings[(size_t)item.string];
    if (item.kind == V_INT) return std::to_string(item.number);
    if (item.kind == V_BOOL) return item.number ? "True" : "False";
    PyErr_SetString(PyExc_ValueError, "guard key of an unexpected value");
    throw Fail();
}

std::string str_of(int32_t value) {  // "" if None else str(value) (guard._command_key)
    const Val &item = V(value);
    if (item.kind == V_NONE) return "";
    return text_or_none(value);
}

Key join(const std::vector<std::string> &parts) {
    Key out;
    for (size_t i = 0; i < parts.size(); i++) { if (i) out.push_back(SEP); out += parts[i]; }
    return out;
}

std::vector<std::string> split(const Key &key) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t found = key.find(SEP, start);
        out.push_back(key.substr(start, found == std::string::npos ? std::string::npos : found - start));
        if (found == std::string::npos) break;
        start = found + 1;
    }
    return out;
}

struct Coordinate { std::string subject; int64_t x, y; bool operator<(const Coordinate &o) const { return std::tie(subject, x, y) < std::tie(o.subject, o.x, o.y); }
                    bool operator==(const Coordinate &o) const { return subject == o.subject && x == o.x && y == o.y; } };
struct SwapKey { std::string left; int64_t a; std::string right; int64_t b;
                 bool operator<(const SwapKey &o) const { return std::tie(left, a, right, b) < std::tie(o.left, o.a, o.right, o.b); }
                 bool operator==(const SwapKey &o) const { return left == o.left && a == o.a && right == o.right && b == o.b; } };

struct Support {
    std::vector<Key> navigation;
    std::vector<std::string> management, movable;
    std::vector<Coordinate> blocked;
    bool has_previous = false;
    Key previous;
    std::vector<SwapKey> swaps;
    std::vector<std::string> windows;
    bool exhausted = false;
};

struct GF {  // fields the guard reads
    int32_t action_kind, action_subject, action_panel, action_selection, action_destination, action_x, action_y, action_count;
    int32_t offers_context, offers_panel, offers_armies, offers_stack_panel, offers_capability_panel, offers_town_refs, offers_dialog_panel;
    int32_t context_panel, context_subject, context_selection, army_subject, army_slots, army_role, army_requires_nonempty;
    int32_t slot_ref, slot_slot, slot_creature, slot_count, hero_position, hero_movement, own_slot_slot, own_slot_creature, own_slot_count;
    int32_t result_accepted, result_status, result_position, result_movement;
    bool ready = false;
} G;

void prepare_guard() {
    if (G.ready) return;
    prepare_types();
    G.action_kind = field_of(T.action, "kind"); G.action_subject = field_of(T.action, "subject"); G.action_panel = field_of(T.action, "panel");
    G.action_selection = field_of(T.action, "selection"); G.action_destination = field_of(T.action, "destination");
    G.action_x = field_of(T.action, "x"); G.action_y = field_of(T.action, "y"); G.action_count = field_of(T.action, "count");
    G.offers_context = field_of(T.offers, "context"); G.offers_panel = field_of(T.offers, "panel"); G.offers_armies = field_of(T.offers, "armies");
    G.offers_stack_panel = field_of(T.offers, "stack_panel"); G.offers_capability_panel = field_of(T.offers, "capability_panel");
    G.offers_town_refs = field_of(T.offers, "town_refs"); G.offers_dialog_panel = field_of(T.offers, "dialog_panel");
    G.context_panel = field_of(T.context, "panel"); G.context_subject = field_of(T.context, "subject"); G.context_selection = field_of(T.context, "selection");
    const int32_t army = type_named("ArmyView"), slot = type_named("ArmySlot"), own = type_named("OwnArmyStack");
    G.army_subject = field_of(army, "subject"); G.army_slots = field_of(army, "slots"); G.army_role = field_of(army, "role");
    G.army_requires_nonempty = field_of(army, "requires_nonempty");
    G.slot_ref = field_of(slot, "ref"); G.slot_slot = field_of(slot, "slot"); G.slot_creature = field_of(slot, "creature"); G.slot_count = field_of(slot, "count");
    G.hero_position = field_of(T.hero, "position"); G.hero_movement = field_of(T.hero, "movement");
    G.own_slot_slot = field_of(own, "slot"); G.own_slot_creature = field_of(own, "creature"); G.own_slot_count = field_of(own, "count");
    G.result_accepted = field_of(T.result, "accepted"); G.result_status = field_of(T.result, "status");
    G.result_position = field_of(T.result, "position"); G.result_movement = field_of(T.result, "movement");
    G.ready = true;
}

inline int64_t num(int32_t value) { return V(value).number; }
inline bool none(int32_t value) { return V(value).kind == V_NONE; }
inline const std::string &str(int32_t value) { return strings[(size_t)V(value).string]; }

bool mandatory(int32_t offers) {  // guard._mandatory
    if (str_is(field(field(offers, G.offers_context), T.c_kind), "dialog")) return true;
    const int32_t panel = field(offers, G.offers_capability_panel);
    return !none(panel) && V(vfield(panel, V(panel).type, "mandatory")).number != 0;
}

const char *NAVIGATION_KINDS[] = {"open_hero", "open_town", "open_panel", "open_building", "open_stack", "open_capability",
                                  "select_recruit", "select_hero", "select_upgrade", "inspect_object"};
const char *MANAGEMENT_KINDS_LIST[] = {"split_stack", "merge_stacks", "swap_stacks", "transfer_stack", "swap_garrison",
                                       "move_artifact", "assemble_artifact", "disassemble_artifact"};

bool in_list(int32_t kind, const char *const *list, size_t count) {
    for (size_t i = 0; i < count; i++) if (str_is(kind, list[i])) return true;
    return false;
}
bool navigation_kind(int32_t kind) { return in_list(kind, NAVIGATION_KINDS, 10); }
bool management_kind(int32_t kind) { return in_list(kind, MANAGEMENT_KINDS_LIST, 8); }

int32_t find_by_ref(int32_t items, int32_t ref) {  // next(item for item in items if item.ref == ref)
    for (int32_t item : V(items).items) if (vfield(item, V(item).type, "ref") == ref) return item;
    PyErr_SetString(PyExc_StopIteration, "no item with that ref");
    throw Fail();
}

bool nav_key(int32_t action, int32_t offers, Key &out) {  // guard.nav_key; false for None
    const int32_t kind = field(action, G.action_kind);
    if (!navigation_kind(kind) || mandatory(offers)) return false;
    const std::string kind_text = str(kind);
    const int32_t subject = field(action, G.action_subject);
    std::vector<std::string> parts = {kind_text, none(subject) ? std::string() : str(subject)};
    if (V(subject).kind == V_STR && str(subject).empty()) parts[1] = "";
    if (kind_text == "open_hero" || kind_text == "open_town") { out = join(parts); return true; }
    if (kind_text == "open_capability") { parts.push_back(text_or_none(field(action, G.action_panel))); out = join(parts); return true; }
    const int32_t panel = field(offers, G.offers_panel);
    const int32_t selection = field(action, G.action_selection);
    if (kind_text == "open_panel") {
        const int32_t which = field(action, G.action_panel);
        parts.push_back(text_or_none(which));
        if (!str_is(which, "recruitment")) { out = join(parts); return true; }
        int index = 0;
        bool found = false;
        for (int32_t control : V(vfield(panel, V(panel).type, "controls")).items) {
            if (!str_is(vfield(control, V(control).type, "panel"), "recruitment")) continue;
            if (vfield(control, V(control).type, "ref") == selection) { found = true; break; }
            index++;
        }
        if (!found) { PyErr_SetString(PyExc_StopIteration, "recruitment control"); throw Fail(); }
        parts.push_back(std::to_string(index));
        out = join(parts);
        return true;
    }
    if (kind_text == "open_building") {
        const int32_t item = find_by_ref(vfield(panel, V(panel).type, "buildings"), selection);
        parts.push_back(text_or_none(vfield(item, V(item).type, "name_text_id")));
    } else if (kind_text == "select_recruit") {
        const int32_t recruitment = vfield(panel, V(panel).type, "recruitment");
        const int32_t item = find_by_ref(vfield(recruitment, V(recruitment).type, "offers"), selection);
        parts.push_back(text_or_none(vfield(item, V(item).type, "creature")));
    } else if (kind_text == "select_hero") {
        const int32_t tavern = vfield(panel, V(panel).type, "tavern");
        const int32_t item = find_by_ref(vfield(tavern, V(tavern).type, "offers"), selection);
        parts.push_back(text_or_none(vfield(item, V(item).type, "name_text_id")));
        parts.push_back(text_or_none(vfield(item, V(item).type, "portrait")));
    } else if (kind_text == "open_stack") {
        int32_t army;
        const int32_t slot = first_slot(field(offers, G.offers_armies), selection, &army);
        if (slot < 0) { PyErr_SetString(PyExc_AttributeError, "open_stack slot"); throw Fail(); }
        parts = {kind_text, str(field(army, G.army_subject)), std::to_string(num(field(slot, G.slot_slot)))};
    } else if (kind_text == "select_upgrade") {
        const int32_t stack = field(offers, G.offers_stack_panel);
        const int32_t item = find_by_ref(vfield(stack, V(stack).type, "offers"), selection);
        const int32_t source = vfield(stack, V(stack).type, "source");
        parts.push_back(text_or_none(vfield(source, V(source).type, "slot")));
        parts.push_back(text_or_none(vfield(item, V(item).type, "creature")));
    } else {
        const int32_t capability = field(offers, G.offers_capability_panel);
        const int32_t item = find_by_ref(vfield(capability, V(capability).type, "choices"), selection);
        parts.push_back(text_or_none(vfield(item, V(item).type, "destination")));
    }
    out = join(parts);
    return true;
}

Key command_key(int32_t action, int32_t offers) {  // guard._command_key
    const int32_t kind = field(action, G.action_kind);
    const std::string kind_text = str(kind);
    const int32_t context = field(offers, G.offers_context);
    std::vector<std::string> parts = {kind_text, str_of(field(context, T.c_kind)), str_of(field(context, G.context_panel)),
                                      str_of(field(context, G.context_subject)), str_of(field(action, G.action_subject))};
    Key navigation;
    const int32_t panel = field(offers, G.offers_panel);
    const int32_t selection = field(action, G.action_selection);
    if (nav_key(action, offers, navigation)) {
        std::vector<std::string> nav = split(navigation);
        for (size_t i = 2; i < nav.size(); i++) parts.push_back(nav[i] == NONE_ELEMENT ? std::string() : nav[i]);
    } else if (kind_text == "move_to") {
        parts.push_back(str_of(field(action, G.action_x)));
        parts.push_back(str_of(field(action, G.action_y)));
    } else if (kind_text == "recruit") {
        const int32_t recruitment = vfield(panel, V(panel).type, "recruitment");
        const int32_t card = find_by_ref(vfield(recruitment, V(recruitment).type, "offers"), selection);
        const int32_t destination = vfield(recruitment, V(recruitment).type, "destination");
        parts.push_back(str_of(vfield(card, V(card).type, "creature")));
        parts.push_back(str_of(vfield(destination, V(destination).type, "role")));
    } else if (kind_text == "construct") {
        const int32_t building = vfield(panel, V(panel).type, "building");
        parts.push_back(str_of(vfield(building, V(building).type, "name_text_id")));
    } else if (kind_text == "hire_hero") {
        const int32_t tavern = vfield(panel, V(panel).type, "tavern");
        const int32_t card = find_by_ref(vfield(tavern, V(tavern).type, "offers"), selection);
        parts.push_back(str_of(vfield(card, V(card).type, "name_text_id")));
        parts.push_back(str_of(vfield(card, V(card).type, "portrait")));
    } else if (kind_text == "trade_resource" || kind_text == "buy_artifact" || kind_text == "move_artifact" || kind_text == "assemble_artifact"
               || kind_text == "disassemble_artifact" || kind_text == "cast_adventure_spell" || kind_text == "choose_level_up"
               || kind_text == "choose_spell_target" || kind_text == "cancel_spell_target" || kind_text == "inspect_object") {  // CAPABILITY_ACTIONS
        const int32_t capability = field(offers, G.offers_capability_panel);
        const int32_t choice = find_by_ref(vfield(capability, V(capability).type, "choices"), selection);
        const int32_t choice_type = V(choice).type;
        for (const char *name : {"artifact", "slot", "destination", "destination_slot", "spell", "skill", "skill_level",
                                 "resource_from", "resource_to", "x", "y"})
            parts.push_back(str_of(vfield(choice, choice_type, name)));
    } else {
        parts.push_back(str_of(selection));
        parts.push_back(str_of(field(action, G.action_destination)));
    }
    parts.push_back(str_of(field(action, G.action_count)));
    return join(parts);
}

bool role_in(int32_t army, const char *a, const char *b) {
    const int32_t role = field(army, G.army_role);
    return str_is(role, a) || str_is(role, b);
}
bool allied_role(int32_t army) { return role_in(army, "allied_garrison", "allied_locked_garrison"); }
bool locked_role(int32_t army) { return role_in(army, "locked_garrison", "allied_locked_garrison"); }
int nonempty_slots(int32_t army) {
    int count = 0;
    for (int32_t slot : V(field(army, G.army_slots)).items) count += num(field(slot, G.slot_count)) != 0;
    return count;
}

int64_t swap_maximum(int32_t source_army, int32_t source, int32_t destination_army, int32_t destination) {  // _operation_maximum("swap_stacks")
    const bool crossing = field(source_army, G.army_subject) != field(destination_army, G.army_subject);
    const int64_t source_count = num(field(source, G.slot_count)), destination_count = num(field(destination, G.slot_count));
    if (source_count && (allied_role(source_army) || (crossing && locked_role(source_army)))) return 0;
    if (destination_count && (allied_role(destination_army) || (crossing && locked_role(destination_army)))) return 0;
    if (!source_count && !destination_count) return 0;
    const bool nonempty_source = V(field(source_army, G.army_requires_nonempty)).number != 0;
    const bool nonempty_destination = V(field(destination_army, G.army_requires_nonempty)).number != 0;
    if (crossing && nonempty_source && source_count && !destination_count && nonempty_slots(source_army) == 1) return 0;
    if (crossing && nonempty_destination && destination_count && !source_count && nonempty_slots(destination_army) == 1) return 0;
    return 1;
}

struct SlotPlace { int32_t army, slot; };

std::unordered_map<int32_t, SlotPlace> slot_places(int32_t armies) {  // army_slot's index: the first slot of each ref
    std::unordered_map<int32_t, SlotPlace> found;
    for (int32_t army : V(armies).items)
        for (int32_t slot : V(field(army, G.army_slots)).items) found.emplace(field(slot, G.slot_ref), SlotPlace{army, slot});
    return found;
}

SwapKey swap_key_of(const SlotPlace &left, const SlotPlace &right) {  // guard._swap_key
    std::pair<std::string, int64_t> first(str(field(left.army, G.army_subject)), num(field(left.slot, G.slot_slot)));
    std::pair<std::string, int64_t> second(str(field(right.army, G.army_subject)), num(field(right.slot, G.slot_slot)));
    if (second < first) std::swap(first, second);
    return SwapKey{first.first, first.second, second.first, second.second};
}

const std::vector<SwapKey> &swap_keys(int32_t armies) {  // guard._swap_keys: a function of the armies value
    static std::unordered_map<int32_t, std::vector<SwapKey>> known;
    auto found = known.find(armies);
    if (found != known.end()) return found->second;
    if (known.size() >= 4096) known.clear();
    std::vector<SwapKey> keys;
    std::set<SwapKey> seen;
    const auto places = slot_places(armies);
    for (int32_t first : V(armies).items)
        for (int32_t source : V(field(first, G.army_slots)).items) {
            const int32_t source_ref = field(source, G.slot_ref);
            const SlotPlace &origin = places.at(source_ref);
            const bool empty = num(field(origin.slot, G.slot_count)) == 0;
            for (int32_t second : V(armies).items)
                for (int32_t destination : V(field(second, G.army_slots)).items) {
                    const int32_t destination_ref = field(destination, G.slot_ref);
                    if (source_ref == destination_ref) continue;
                    const SlotPlace &target = places.at(destination_ref);
                    if (empty && num(field(target.slot, G.slot_count)) == 0) continue;
                    if (swap_maximum(origin.army, origin.slot, target.army, target.slot) <= 0) continue;
                    SwapKey key = swap_key_of(origin, target);
                    if (seen.insert(key).second) keys.push_back(key);
                }
        }
    return known.emplace(armies, std::move(keys)).first->second;
}

typedef std::vector<std::pair<int32_t, int64_t>> LayoutRow;  // (creature value, count) per slot
typedef std::vector<std::pair<std::string, LayoutRow>> Layout;  // sorted by subject

Layout army_layout(int32_t observation, int32_t armies) {  // guard._army_layout
    std::map<std::string, LayoutRow> rows;
    const int32_t own = field(observation, T.o_own_armies);
    if (!none(own))
        for (int32_t army : V(own).items) {
            LayoutRow slots(7, std::make_pair(v_none, (int64_t)0));
            const int32_t type = V(army).type;
            for (int32_t slot : V(vfield(army, type, "slots")).items) {
                const int64_t index = num(field(slot, G.own_slot_slot));
                if (index < -7 || index >= 7) { PyErr_SetString(PyExc_IndexError, "own army slot"); throw Fail(); }
                slots[(size_t)(index < 0 ? index + 7 : index)] = {field(slot, G.own_slot_creature), num(field(slot, G.own_slot_count))};
            }
            rows[str(vfield(army, type, "subject"))] = std::move(slots);
        }
    if (!none(armies))
        for (int32_t army : V(armies).items) {
            LayoutRow slots;
            for (int32_t slot : V(field(army, G.army_slots)).items) slots.push_back({field(slot, G.slot_creature), num(field(slot, G.slot_count))});
            rows[str(field(army, G.army_subject))] = std::move(slots);
        }
    return Layout(rows.begin(), rows.end());
}

std::vector<SwapKey> blocked_swaps(int32_t observation, int32_t offers, const std::set<Layout> &seen) {  // guard._blocked_swaps
    const int32_t armies = field(offers, G.offers_armies);
    std::vector<SwapKey> out;
    if (none(armies) || V(armies).items.empty()) return out;
    const Layout layout = army_layout(observation, armies);
    std::unordered_map<std::string, size_t> places;
    for (size_t i = 0; i < layout.size(); i++) places[layout[i].first] = i;
    for (const SwapKey &key : swap_keys(armies)) {
        const size_t i = places.at(key.left), j = places.at(key.right);
        Layout rows = layout;
        auto at = [](LayoutRow &row, int64_t index) -> std::pair<int32_t, int64_t> & {
            const int64_t size = (int64_t)row.size();
            if (index < -size || index >= size) { PyErr_SetString(PyExc_IndexError, "layout slot"); throw Fail(); }
            return row[(size_t)(index < 0 ? index + size : index)];
        };
        const auto x = at(rows[i].second, key.a), y = at(rows[j].second, key.b);
        at(rows[i].second, key.a) = y;
        at(rows[j].second, key.b) = x;
        if (seen.count(rows)) out.push_back(key);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

std::vector<std::string> management_subjects(int32_t action, int32_t offers) {  // guard.management_subjects
    const int32_t kind = field(action, G.action_kind);
    std::vector<std::string> out;
    if (!management_kind(kind)) return out;
    std::set<std::string> subjects;
    auto subject_text = [](int32_t value) { return none(value) ? NONE_ELEMENT : str(value); };
    if (str_is(kind, "move_artifact") || str_is(kind, "assemble_artifact") || str_is(kind, "disassemble_artifact")) {
        const int32_t capability = field(offers, G.offers_capability_panel);
        const int32_t choice = find_by_ref(vfield(capability, V(capability).type, "choices"), field(action, G.action_selection));
        subjects.insert(subject_text(vfield(choice, V(choice).type, "subject")));
        const int32_t destination = vfield(choice, V(choice).type, "destination");
        if (!none(destination)) subjects.insert(str(destination));
    } else if (str_is(kind, "swap_garrison")) {
        subjects.insert(subject_text(field(action, G.action_subject)));
        const int32_t armies = field(offers, G.offers_armies);
        if (!none(armies)) for (int32_t army : V(armies).items) subjects.insert(str(field(army, G.army_subject)));
    } else {
        int32_t armies = field(offers, G.offers_armies);
        std::vector<int32_t> list;
        if (!none(armies)) list = V(armies).items;
        const int32_t panel = field(offers, G.offers_panel);
        if (str_is(kind, "transfer_stack") && !none(panel)) {
            const int32_t exchange = vfield(panel, V(panel).type, "armies");
            if (!none(exchange)) {
                list = {vfield(exchange, V(exchange).type, "source")};
                const int32_t destination = vfield(exchange, V(exchange).type, "destination");
                if (!none(destination)) list.push_back(destination);
            }
        }
        for (int32_t ref : {field(action, G.action_selection), field(action, G.action_destination)})
            for (int32_t army : list) {
                bool found = false;
                for (int32_t slot : V(field(army, G.army_slots)).items) if (field(slot, G.slot_ref) == ref) { found = true; break; }
                if (found) { subjects.insert(str(field(army, G.army_subject))); break; }
            }
    }
    if (subjects.empty()) subjects.insert(subject_text(field(action, G.action_subject)));
    return std::vector<std::string>(subjects.begin(), subjects.end());
}

// A component of the world, sorted as guard._build_world sorts it, interned: equal components, equal ids.
template <class KeyOf> int32_t sorted_component(int32_t tuple, std::unordered_map<int32_t, int32_t> &known, KeyOf key_of) {
    auto found = known.find(tuple);
    if (found != known.end()) return found->second;
    if (known.size() >= 65536) known.clear();
    const std::vector<int32_t> &items = items_of(tuple);
    std::vector<std::pair<decltype(key_of(0)), size_t>> keyed;
    keyed.reserve(items.size());
    for (size_t i = 0; i < items.size(); i++) keyed.push_back({key_of(items[i]), i});
    std::stable_sort(keyed.begin(), keyed.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    std::vector<int32_t> sorted;
    sorted.reserve(items.size());
    for (auto &entry : keyed) sorted.push_back(items[entry.second]);
    return known[tuple] = v_items(V_TUPLE, -1, std::move(sorted));
}

struct World { int32_t parts[6]; bool operator==(const World &o) const { return std::equal(parts, parts + 6, o.parts); } bool operator!=(const World &o) const { return !(*this == o); } };

World world_of(int32_t observation) {  // guard._world, as interned components
    static std::unordered_map<int32_t, int32_t> tiles, objects, heroes, towns;
    static std::unordered_map<int32_t, World> known;
    auto found = known.find(observation);
    if (found != known.end()) return found->second;
    if (known.size() >= 65536) known.clear();
    World world;
    world.parts[0] = field(observation, T.o_size);
    world.parts[1] = sorted_component(field(observation, T.o_tiles), tiles, [](int32_t tile) {
        const auto &position = items_of(field(tile, T.tile_position));
        return std::make_tuple(num(position[0]), num(position[1]), num(position[2])); });
    world.parts[2] = sorted_component(field(observation, T.o_objects), objects, [](int32_t item) { return str(F(item, "handle")); });
    world.parts[3] = F(observation, "resources");
    const int32_t hero_tuple = field(observation, T.o_heroes);
    auto known_heroes = heroes.find(hero_tuple);
    if (known_heroes != heroes.end()) world.parts[4] = known_heroes->second;
    else {
        std::vector<std::pair<std::string, size_t>> keyed;
        const auto &items = items_of(hero_tuple);
        for (size_t i = 0; i < items.size(); i++) keyed.push_back({str(field(items[i], T.h_handle)), i});
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        std::vector<int32_t> flat;
        for (auto &entry : keyed) {
            const int32_t hero = items[entry.second];
            flat.push_back(field(hero, T.h_handle)); flat.push_back(field(hero, G.hero_position)); flat.push_back(field(hero, G.hero_movement));
        }
        if (heroes.size() >= 65536) heroes.clear();
        world.parts[4] = heroes[hero_tuple] = v_items(V_TUPLE, -1, std::move(flat));
    }
    world.parts[5] = sorted_component(field(observation, T.o_towns), towns, [](int32_t town) { return str(field(town, T.t_handle)); });
    return known[observation] = world;
}

struct Tracker {
    int64_t management_limit = 32, window_limit = -1, decision_limit = -1;
    std::map<std::string, int64_t> windows, management;
    std::map<Key, int64_t> navigation;
    int64_t decisions = 0;
    bool has_day = false;
    int32_t day = 0, last_seq = -1;
    bool has_world = false;
    World world;
    std::set<Coordinate> failed;
    std::set<Layout> layouts;
    bool town_reopen_blocked = false, has_previous = false, has_move = false, consumed = false;
    Key previous;
    Coordinate previous_move;
    int32_t decision_observation = -1, decision_offers = -1, action = -1;
    int64_t epoch = 0, action_epoch = -1;

    void progress() { navigation.clear(); failed.clear(); layouts.clear(); epoch++; }

    bool observe(const NFrame &frame, Support &out) {  // Tracker.observe; true at a decision, with its support
        if (frame.seq <= last_seq) { PyErr_SetString(PyExc_ValueError, "Interaction guard requires one ordered player episode"); throw Fail(); }
        last_seq = frame.seq;
        if (!has_day || day != frame.day) {
            if (has_day && frame.day < day) { PyErr_SetString(PyExc_ValueError, "Interaction guard day cannot move backwards"); throw Fail(); }
            has_day = true; day = frame.day;
            management.clear(); windows.clear(); decisions = 0;
            progress();
        }
        if (frame.observation >= 0) {
            const World now = world_of(frame.observation);
            const bool managing = consumed && action >= 0 && management_kind(field(action, G.action_kind));
            if (has_world && world != now && !managing) progress();
            world = now; has_world = true;
        }
        if (frame.kind == F_DECISION) {
            decision_observation = frame.observation; decision_offers = frame.offers; action = -1; consumed = false;
            const int32_t observation = frame.observation, offers = frame.offers;
            out = Support();
            std::set<Coordinate> blocked = failed;
            for (int32_t hero : V(field(observation, T.o_heroes)).items) {
                const std::string handle = str(field(hero, T.h_handle));
                if (num(field(hero, G.hero_movement)) > 0 && !V(field(hero, T.h_in_garrison)).number) out.movable.push_back(handle);
                const auto &position = V(field(hero, G.hero_position)).items;
                blocked.insert(Coordinate{handle, num(position[0]), num(position[1])});
            }
            std::sort(out.movable.begin(), out.movable.end());
            if (has_move) blocked.insert(previous_move);
            out.blocked.assign(blocked.begin(), blocked.end());
            std::set<Key> nav;
            for (auto &entry : navigation) if (entry.second >= 2) nav.insert(entry.first);
            if (town_reopen_blocked)
                for (int32_t subject : V(field(offers, G.offers_town_refs)).items) nav.insert(join({"open_town", str(subject)}));
            out.navigation.assign(nav.begin(), nav.end());
            for (auto &entry : management) if (entry.second >= management_limit) out.management.push_back(entry.first);
            out.has_previous = has_previous; out.previous = previous;
            const int32_t armies = field(offers, G.offers_armies);
            layouts.insert(army_layout(observation, armies));
            out.swaps = blocked_swaps(observation, offers, layouts);
            if (window_limit >= 0) for (auto &entry : windows) if (entry.second >= window_limit) out.windows.push_back(entry.first);
            out.exhausted = decision_limit >= 0 && decisions >= decision_limit;
            return true;
        }
        if (frame.kind == F_SUBMITTED) {
            action = frame.action;
        } else if (frame.kind == F_CONSUMED) {
            if (decision_observation < 0 || action < 0 || consumed) {
                PyErr_SetString(PyExc_ValueError, "Interaction guard requires one submitted action per consumption"); throw Fail();
            }
            consumed = true;
            action_epoch = epoch;
            has_previous = true;
            previous = command_key(action, decision_offers);
            has_move = str_is(field(action, G.action_kind), "move_to");
            if (has_move) previous_move = Coordinate{str(field(action, G.action_subject)), num(field(action, G.action_x)), num(field(action, G.action_y))};
            Key key;
            if (nav_key(action, decision_offers, key)) {
                navigation[key]++;
                windows[split(key)[1]]++;
            }
            const bool forced = mandatory(decision_offers);
            decisions += !forced;
            if (!forced) for (const std::string &subject : management_subjects(action, decision_offers)) management[subject]++;
        } else if (frame.kind == F_EVENT) {
            const int32_t kind = field(frame.event, T.event_kind);
            if (str_is(kind, "battle_start") || str_is(kind, "battle_complete") || str_is(kind, "hero_lost") || str_is(kind, "level_gained")) progress();
        } else if (frame.kind == F_RESULT && action >= 0 && consumed) {
            const int32_t result = frame.result;
            const bool accepted = V(field(result, G.result_accepted)).number != 0;
            const int32_t status = field(result, G.result_status);
            const int32_t kind = field(action, G.action_kind);
            if (accepted)
                town_reopen_blocked = str_is(kind, "close") && str_is(status, "closed")
                                      && str_is(field(field(decision_offers, G.offers_context), T.c_kind), "town");
            if (accepted && (str_is(status, "built") || str_is(status, "recruited") || str_is(status, "hired") || str_is(status, "upgraded")
                             || (str_is(status, "capability_applied") && (str_is(kind, "trade_resource") || str_is(kind, "buy_artifact")
                                                                           || str_is(kind, "choose_level_up")))))
                progress();
            if (str_is(kind, "move_to") && (str_is(status, "no_route_known") || str_is(status, "movement_limit") || str_is(status, "hero_unavailable")
                                            || str_is(status, "reached") || str_is(status, "rejected"))) {
                const int32_t subject = field(action, G.action_subject);
                int32_t hero = -1;
                for (int32_t item : V(field(decision_observation, T.o_heroes)).items) if (field(item, T.h_handle) == subject) { hero = item; break; }
                const int32_t position = field(result, G.result_position), movement = field(result, G.result_movement);
                const bool unchanged = hero >= 0 && (none(position) || position == field(hero, G.hero_position))
                                       && (none(movement) || movement == field(hero, G.hero_movement));
                if (unchanged && action_epoch == epoch)
                    failed.insert(Coordinate{str(subject), num(field(action, G.action_x)), num(field(action, G.action_y))});
            }
        }
        return false;
    }
};

PyObject *key_tuple(const Key &key) {
    std::vector<std::string> parts = split(key);
    Ref out(PyTuple_New((Py_ssize_t)parts.size()));
    for (size_t i = 0; i < parts.size(); i++) {
        PyObject *item = parts[i] == NONE_ELEMENT ? (Py_INCREF(Py_None), Py_None) : PyUnicode_FromStringAndSize(parts[i].data(), (Py_ssize_t)parts[i].size());
        if (!item) throw Fail();
        PyTuple_SET_ITEM(out.get(), (Py_ssize_t)i, item);
    }
    Py_INCREF(out.get());
    return out.get();
}

PyObject *text_tuple(const std::vector<std::string> &items) {
    Ref out(PyTuple_New((Py_ssize_t)items.size()));
    for (size_t i = 0; i < items.size(); i++) {
        PyObject *item = items[i] == NONE_ELEMENT ? (Py_INCREF(Py_None), Py_None) : PyUnicode_FromStringAndSize(items[i].data(), (Py_ssize_t)items[i].size());
        if (!item) throw Fail();
        PyTuple_SET_ITEM(out.get(), (Py_ssize_t)i, item);
    }
    Py_INCREF(out.get());
    return out.get();
}

PyObject *support_tuple(const Support &support) {  // the GuardSupport fields, in order
    Ref navigation(PyTuple_New((Py_ssize_t)support.navigation.size()));
    for (size_t i = 0; i < support.navigation.size(); i++) PyTuple_SET_ITEM(navigation.get(), (Py_ssize_t)i, key_tuple(support.navigation[i]));
    Ref blocked(PyTuple_New((Py_ssize_t)support.blocked.size()));
    for (size_t i = 0; i < support.blocked.size(); i++) {
        const Coordinate &item = support.blocked[i];
        PyTuple_SET_ITEM(blocked.get(), (Py_ssize_t)i, Py_BuildValue("(s#LL)", item.subject.data(), (Py_ssize_t)item.subject.size(), (long long)item.x, (long long)item.y));
    }
    Ref swaps(PyTuple_New((Py_ssize_t)support.swaps.size()));
    for (size_t i = 0; i < support.swaps.size(); i++) {
        const SwapKey &item = support.swaps[i];
        PyTuple_SET_ITEM(swaps.get(), (Py_ssize_t)i, Py_BuildValue("(s#Ls#L)", item.left.data(), (Py_ssize_t)item.left.size(), (long long)item.a,
                                                                   item.right.data(), (Py_ssize_t)item.right.size(), (long long)item.b));
    }
    Ref management(text_tuple(support.management)), movable(text_tuple(support.movable)), windows(text_tuple(support.windows));
    Ref previous(support.has_previous ? key_tuple(support.previous) : (Py_INCREF(Py_None), Py_None));
    return Py_BuildValue("(OOOOOOOO)", navigation.get(), management.get(), movable.get(), blocked.get(), previous.get(), swaps.get(),
                         windows.get(), support.exhausted ? Py_True : Py_False);
}

PyObject *py_guard_supports(PyObject *, PyObject *args) {  // guard_supports(producer, management, windows, decisions) (tests)
    PyObject *object;
    long long management_limit, window_limit, decision_limit;
    if (!PyArg_ParseTuple(args, "OLLL", &object, &management_limit, &window_limit, &decision_limit)) return nullptr;
    if (Py_TYPE(object) != &ProducerType) { PyErr_SetString(PyExc_TypeError, "a Producer"); return nullptr; }
    Producer *producer = (Producer *)object;
    try {
        prepare_guard();
        Tracker tracker;
        tracker.management_limit = management_limit; tracker.window_limit = window_limit; tracker.decision_limit = decision_limit;
        Ref out(PyList_New(0));
        Support support;
        for (const NFrame &frame : *producer->frames) {
            if (!tracker.observe(frame, support)) continue;
            Ref item(support_tuple(support));
            if (!item.get() || PyList_Append(out.get(), item.get())) throw Fail();
        }
        Py_INCREF(out.get());
        return out.get();
    } catch (Fail &) {
        return nullptr;
    }
}

// == window memory (heroes.game.window_repeats) ===================================================================

bool binding_field(const std::string &name) { return name == "ref" || (name.size() >= 4 && name.compare(name.size() - 4, 4, "_ref") == 0); }

const std::vector<int32_t> &bound_refs(int32_t value) {  // the strings _same_window binds in a value compared with itself
    static std::unordered_map<int32_t, std::vector<int32_t>> known;
    auto found = known.find(value);
    if (found != known.end()) return found->second;
    std::vector<int32_t> out;
    const Val &x = V(value);
    if (x.kind == V_TUPLE) {
        for (int32_t item : x.items) { const auto &inner = bound_refs(item); out.insert(out.end(), inner.begin(), inner.end()); }
    } else if (x.kind == V_RECORD) {
        const VType &type = vtypes[(size_t)x.type];
        const bool context = x.type == T.context;
        const bool exchange = context && str_is(x.items[(size_t)T.c_kind], "exchange");
        for (size_t i = 0; i < type.fields.size(); i++) {
            const std::string &name = type.fields[i].name;
            const int32_t item = x.items[i];
            if ((binding_field(name) || (name == "selection" && context && !exchange)) && V(item).kind == V_STR) out.push_back(item);
            else { const auto &inner = bound_refs(item); out.insert(out.end(), inner.begin(), inner.end()); }
        }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (known.size() >= (1u << 20)) known.clear();
    return known.emplace(value, std::move(out)).first->second;
}

struct SameWindow {  // window_repeats._same_window's equal: equal but for a consistent renaming of control refs
    std::unordered_map<int32_t, int32_t> forward, reverse;
    bool bind(int32_t left, int32_t right) {
        auto f = forward.emplace(left, right).first;
        auto r = reverse.emplace(right, left).first;
        return f->second == right && r->second == left;
    }
    bool equal(int32_t a, int32_t b) {
        if (a == b) {  // Python walks it anyway and binds each of its refs to itself
            for (int32_t ref : bound_refs(a)) if (!bind(ref, ref)) return false;
            return true;
        }
        const Val &x = V(a), &y = V(b);
        if (x.kind != y.kind) return false;
        if (x.kind == V_TUPLE) {
            if (x.items.size() != y.items.size()) return false;
            for (size_t i = 0; i < x.items.size(); i++) if (!equal(x.items[i], y.items[i])) return false;
            return true;
        }
        if (x.kind != V_RECORD || x.type != y.type) return false;  // unequal scalars, or two types
        const VType &type = vtypes[(size_t)x.type];
        const bool context = x.type == T.context;
        const bool exchange = context && str_is(x.items[(size_t)T.c_kind], "exchange");
        for (size_t i = 0; i < type.fields.size(); i++) {
            const std::string &name = type.fields[i].name;
            const int32_t left = x.items[i], right = y.items[i];
            if ((binding_field(name) || (name == "selection" && context && !exchange)) && V(left).kind == V_STR && V(right).kind == V_STR) {
                if (!bind(left, right)) return false;
            } else if (!equal(left, right)) {
                return false;
            }
        }
        return true;
    }
};

bool same_window(int32_t left, int32_t right) { return left == right || SameWindow().equal(left, right); }

const std::vector<int32_t> &members(int32_t tuple) {  // set(tuple), as sorted distinct value ids
    static std::unordered_map<int32_t, std::vector<int32_t>> known;
    auto found = known.find(tuple);
    if (found != known.end()) return found->second;
    std::vector<int32_t> out = V(tuple).items;
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    if (known.size() >= 65536) known.clear();
    return known.emplace(tuple, std::move(out)).first->second;
}

bool same_members(int32_t left, int32_t right) {
    return left == right || (!none(left) && !none(right) && members(left) == members(right));
}

struct WF { int32_t o_resources, o_army_information, context_panel, result_accepted, result_status; bool ready = false; } WFI;

bool same_world(int32_t a, int32_t b) {  // window_repeats._same_world
    if (a == b) return true;
    if (!WFI.ready) {
        WFI.o_resources = field_of(T.observation, "resources"); WFI.o_army_information = field_of(T.observation, "army_information");
        WFI.ready = true;
    }
    return field(a, T.o_size) == field(b, T.o_size) && field(a, WFI.o_resources) == field(b, WFI.o_resources)
           && field(a, T.o_heroes) == field(b, T.o_heroes) && field(a, T.o_towns) == field(b, T.o_towns)
           && field(a, WFI.o_army_information) == field(b, WFI.o_army_information)
           && same_members(field(a, T.o_own_armies), field(b, T.o_own_armies)) && same_members(field(a, T.o_tiles), field(b, T.o_tiles))
           && same_members(field(a, T.o_objects), field(b, T.o_objects));
}

int32_t observation_context(int32_t observation) {  // Observation.context
    const int32_t window = field(observation, T.o_window);
    return none(window) ? default_context() : field(window, T.w_context);
}

struct Step { int32_t before = -1, after = -1; bool has_target = false; Key target; };

const char *OPEN_KINDS[] = {"open_hero", "open_town", "open_panel", "open_building", "open_stack", "open_capability"};
const char *CLOSE_KINDS[] = {"close", "back", "close_capability"};

bool same_view(const NFrame &left, const NFrame &right) {  // window_repeats._same_view
    return left.day == right.day && same_world(left.observation, right.observation)
           && same_window(field(left.observation, T.o_window), field(right.observation, T.o_window));
}

// window_repeats._navigation_step over frames[order[start]] .. ; ``order`` lists frame indices
bool navigation_step(const std::vector<NFrame> &all, const std::vector<int32_t> &order, size_t start, size_t stop, Step &out) {
    const NFrame &before = all[(size_t)order[start]];
    int32_t action = -1, after = -1;
    int32_t context = observation_context(before.observation);
    for (size_t i = start + 1; i < stop; i++) {
        const NFrame &frame = all[(size_t)order[i]];
        if (frame.kind == F_SUBMITTED) {
            if (frame.decision_seq != before.seq) return false;
            action = frame.action;
            const int32_t kind = field(action, G.action_kind);
            if (!in_list(kind, OPEN_KINDS, 6) && !in_list(kind, CLOSE_KINDS, 3)) return false;
        } else if (frame.kind == F_CONTEXT) {
            context = frame.context;
        } else if (frame.kind == F_VIEW) {
            after = order[i];
        } else if (frame.kind == F_RESULT) {
            if (action < 0 || frame.decision_seq != before.seq || !V(field(frame.result, G.result_accepted)).number) return false;
            const int32_t kind = field(action, G.action_kind), status = field(frame.result, G.result_status);
            Key target;
            bool has_target = false;
            if (in_list(kind, OPEN_KINDS, 6) && str_is(status, "opened")) {
                if (!nav_key(action, before.offers, target)) return false;
                has_target = true;
            } else if (!(in_list(kind, CLOSE_KINDS, 3) && str_is(status, "closed"))) {
                return false;
            }
            if (after < 0 || observation_context(all[(size_t)after].observation) != context) {
                after = -1;
                const size_t limit = std::min(stop + 1, order.size());
                for (size_t j = i + 1; j < limit; j++) {
                    const FKind k = all[(size_t)order[j]].kind;
                    if (k == F_VIEW || k == F_DECISION) { after = order[j]; break; }
                }
            }
            if (after >= 0 && observation_context(all[(size_t)after].observation) == context) {
                out.before = order[start]; out.after = after; out.has_target = has_target; out.target = target;
                return true;
            }
            return false;
        }
    }
    return false;
}

struct WindowMemory {
    bool has_day = false;
    int32_t day = 0, observation = -1;
    std::map<std::vector<int32_t>, std::pair<int32_t, std::set<Key>>> parents;  // (kind, subject, panel) or {} -> (window, keys)
    std::vector<int32_t> frames;  // frame indices: one decision through its successor
    bool has_opened = false;
    Step opened;

    std::vector<int32_t> parent_of(int32_t window) {
        if (none(window)) return {};
        const int32_t context = field(window, T.w_context);
        return {field(context, T.c_kind), field(context, G.context_subject), field(context, G.context_panel)};
    }

    std::set<Key> at(int32_t observation_value, std::vector<int32_t> *parent_out) {  // WindowMemory._at
        const int32_t window = field(observation_value, T.o_window);
        std::vector<int32_t> parent = parent_of(window);
        if (parent_out) *parent_out = parent;
        auto found = parents.find(parent);
        if (found != parents.end()) {
            if (same_window(found->second.first, window)) return found->second.second;
            parents.erase(found);
        }
        return {};
    }

    // WindowMemory.observe: true at a decision, with its keys
    bool observe(const std::vector<NFrame> &all, int32_t index, std::set<Key> &out) {
        const NFrame &frame = all[(size_t)index];
        const bool changed_world = frame.observation >= 0 && observation >= 0 && !same_world(observation, frame.observation);
        bool changed_by_action = false;
        if (frame.result >= 0 && V(field(frame.result, G.result_accepted)).number) {
            const int32_t status = field(frame.result, G.result_status);
            for (const char *name : {"built", "recruited", "hired", "upgraded", "transferred", "split", "merged", "swapped",
                                     "garrison_swapped", "capability_applied", "dialog_answered", "interaction"})
                if (str_is(status, name)) { changed_by_action = true; break; }
        }
        if (!has_day || frame.day != day || changed_world || changed_by_action || frame.kind == F_EVENT) {
            parents.clear(); frames.clear(); has_opened = false;
        }
        has_day = true; day = frame.day;
        if (frame.observation >= 0) observation = frame.observation;
        if (!frames.empty()) frames.push_back(index);
        if (frame.kind != F_DECISION) return false;
        Step step;
        const bool stepped = !frames.empty() && navigation_step(all, frames, 0, frames.size() - 1, step);
        frames.assign(1, index);
        const bool had_previous = has_opened;
        const Step previous = opened;
        has_opened = stepped && step.has_target;
        if (has_opened) opened = step;
        if (has_opened) {
            const int32_t source = all[(size_t)step.before].observation;
            std::vector<int32_t> parent;
            std::set<Key> keys = at(source, &parent);
            if (keys.count(step.target)) {
                keys.erase(step.target);
                parents[parent] = {field(source, T.o_window), keys};
            }
        } else if (stepped && had_previous && same_view(all[(size_t)previous.after], all[(size_t)step.before])) {
            if (same_view(all[(size_t)previous.before], all[(size_t)step.after]) && same_view(all[(size_t)step.after], frame)) {
                std::vector<int32_t> parent;
                std::set<Key> keys = at(frame.observation, &parent);
                keys.insert(previous.target);
                parents[parent] = {field(frame.observation, T.o_window), keys};
            }
        }
        out = at(frame.observation, nullptr);
        return true;
    }
};

PyObject *py_window_memories(PyObject *, PyObject *object) {  // window_memories(producer): each decision's keys (tests)
    if (Py_TYPE(object) != &ProducerType) { PyErr_SetString(PyExc_TypeError, "a Producer"); return nullptr; }
    Producer *producer = (Producer *)object;
    try {
        prepare_guard();
        WindowMemory memory;
        Ref out(PyList_New(0));
        std::set<Key> keys;
        for (size_t i = 0; i < producer->frames->size(); i++) {
            if (!memory.observe(*producer->frames, (int32_t)i, keys)) continue;
            Ref set(PyFrozenSet_New(nullptr));
            for (const Key &key : keys) { Ref item(key_tuple(key)); if (PySet_Add(set.get(), item.get())) throw Fail(); }
            if (PyList_Append(out.get(), set.get())) throw Fail();
        }
        Py_INCREF(out.get());
        return out.get();
    } catch (Fail &) {
        return nullptr;
    }
}

// == the action walk (policy.ContextualDistribution._walk) ========================================================

inline bool truthy_tuple(int32_t value) { return V(value).kind == V_TUPLE && !V(value).items.empty(); }
inline int32_t int_value(int64_t number) { return v_scalar(V_INT, number); }
int32_t text_value(const char *text) { return str_value(std::string(text)); }

struct ActionSpec { int32_t kind, subject, x, y, panel, selection, count, destination; };

struct AF { int32_t kind, subject, x, y, schema, panel, selection, count, destination; size_t size; bool ready = false; } AFI;

void prepare_actions() {
    if (!AFI.ready) {
        AFI.kind = field_of(T.action, "kind"); AFI.subject = field_of(T.action, "subject"); AFI.x = field_of(T.action, "x");
        AFI.y = field_of(T.action, "y"); AFI.schema = field_of(T.action, "schema"); AFI.panel = field_of(T.action, "panel");
        AFI.selection = field_of(T.action, "selection"); AFI.count = field_of(T.action, "count"); AFI.destination = field_of(T.action, "destination");
        AFI.size = vtypes[(size_t)T.action].fields.size();
        if (AFI.size != 9) { PyErr_SetString(PyExc_ValueError, "Action has fields native.cpp does not know"); throw Fail(); }
        AFI.ready = true;
    }
}

int32_t make_action(const ActionSpec &spec, int32_t schema) {
    prepare_actions();
    std::vector<int32_t> items(AFI.size);
    items[(size_t)AFI.kind] = spec.kind; items[(size_t)AFI.subject] = spec.subject; items[(size_t)AFI.x] = spec.x; items[(size_t)AFI.y] = spec.y;
    items[(size_t)AFI.schema] = schema; items[(size_t)AFI.panel] = spec.panel; items[(size_t)AFI.selection] = spec.selection;
    items[(size_t)AFI.count] = spec.count; items[(size_t)AFI.destination] = spec.destination;
    return v_items(V_RECORD, T.action, std::move(items));
}

int64_t operation_maximum(const std::string &kind, const SlotPlace &origin, const SlotPlace &target) {  // contract._operation_maximum
    if (kind == "swap_stacks") return swap_maximum(origin.army, origin.slot, target.army, target.slot);
    const int32_t source_army = origin.army, source = origin.slot, destination_army = target.army, destination = target.slot;
    if (kind == "split_stack" && str_is(field(source_army, G.army_role), "joining_neutral")) return 0;
    const bool crossing = field(source_army, G.army_subject) != field(destination_army, G.army_subject);
    if (allied_role(source_army) || (crossing && locked_role(source_army))) return 0;
    const int64_t source_count = num(field(source, G.slot_count)), destination_count = num(field(destination, G.slot_count));
    if ((kind != "split_stack" && kind != "merge_stacks") || !source_count) return 0;
    const int32_t creature = field(destination, G.slot_creature);
    if (!none(creature) && creature != field(source, G.slot_creature)) return 0;
    if (source_count + destination_count > 2147483647LL) return 0;
    const int64_t reserve = crossing && V(field(source_army, G.army_requires_nonempty)).number && nonempty_slots(source_army) == 1;
    const int64_t maximum = source_count - reserve;
    if (kind == "merge_stacks") return destination_count != 0 && maximum == source_count;
    return maximum;
}

struct Pair { int32_t source, destination; int64_t maximum; };

const std::vector<Pair> &operation_choices(int32_t armies, const std::string &kind) {  // contract.army_operation_choices
    static std::map<std::pair<int32_t, std::string>, std::vector<Pair>> known;
    auto key = std::make_pair(armies, kind);
    auto found = known.find(key);
    if (found != known.end()) return found->second;
    if (known.size() >= 4096) known.clear();
    std::vector<Pair> out;
    const auto places = slot_places(armies);
    for (int32_t first : items_of(armies))
        for (int32_t source : items_of(field(first, G.army_slots))) {
            const int32_t source_ref = field(source, G.slot_ref);
            const SlotPlace &origin = places.at(source_ref);
            for (int32_t second : items_of(armies))
                for (int32_t destination : items_of(field(second, G.army_slots))) {
                    const int32_t destination_ref = field(destination, G.slot_ref);
                    if (source_ref == destination_ref) continue;
                    const int64_t maximum = operation_maximum(kind, origin, places.at(destination_ref));
                    if (maximum > 0) out.push_back(Pair{source_ref, destination_ref, maximum});
                }
        }
    return known.emplace(key, std::move(out)).first->second;
}

int64_t army_operation_maximum(int32_t armies, const std::string &kind, int32_t source_ref, int32_t destination_ref) {
    const auto places = slot_places(armies);
    auto source = places.find(source_ref), destination = places.find(destination_ref);
    if (source == places.end() || destination == places.end() || source_ref == destination_ref) return 0;
    return operation_maximum(kind, source->second, destination->second);
}

int32_t slot_with_ref(int32_t army, int32_t ref) {
    for (int32_t slot : items_of(field(army, G.army_slots))) if (field(slot, G.slot_ref) == ref) return slot;
    return -1;
}

int64_t exchange_maximum(int32_t exchange, int32_t source_ref, int32_t destination_ref) {  // ArmyExchange.maximum
    const int32_t destination_army = F(exchange, "destination");
    if (none(destination_army)) return 0;
    const int32_t source_army = F(exchange, "source");
    const int32_t source = slot_with_ref(source_army, source_ref), destination = slot_with_ref(destination_army, destination_ref);
    if (source < 0 || destination < 0) return 0;
    const int64_t source_count = num(field(source, G.slot_count)), destination_count = num(field(destination, G.slot_count));
    if (source_count == 0) return 0;
    const int32_t creature = field(destination, G.slot_creature);
    if (!none(creature) && creature != field(source, G.slot_creature)) return 0;
    if (source_count + destination_count > 2147483647LL) return 0;
    const int64_t reserve = V(field(source_army, G.army_requires_nonempty)).number && nonempty_slots(source_army) == 1;
    return source_count - reserve;
}

std::vector<Pair> exchange_choices(int32_t exchange) {  // ArmyExchange.choices
    std::vector<Pair> out;
    const int32_t destination_army = F(exchange, "destination");
    if (none(destination_army)) return out;
    for (int32_t source : items_of(field(F(exchange, "source"), G.army_slots)))
        for (int32_t destination : items_of(field(destination_army, G.army_slots))) {
            const int64_t maximum = exchange_maximum(exchange, field(source, G.slot_ref), field(destination, G.slot_ref));
            if (maximum > 0) out.push_back(Pair{field(source, G.slot_ref), field(destination, G.slot_ref), maximum});
        }
    return out;
}

bool contains(int32_t tuple, int32_t value) {
    if (none(tuple)) return false;
    for (int32_t item : items_of(tuple)) if (item == value) return true;
    return false;
}
bool ref_in(int32_t items, int32_t ref) {  // ref in {item.ref for item in items}
    for (int32_t item : items_of(items)) if (F(item, "ref") == ref) return true;
    return false;
}
int64_t count_of(int32_t action) {
    const int32_t count = field(action, G.action_count);
    if (none(count)) { PyErr_SetString(PyExc_TypeError, "'<=' not supported between instances of 'NoneType' and 'int'"); throw Fail(); }
    return num(count);
}

bool offers_permit(int32_t offers, int32_t action) {  // Offers.permits
    if (field(action, AFI.schema) != F(offers, "schema")) return false;
    const int32_t context = field(offers, G.offers_context);
    const int32_t subject = field(action, G.action_subject), kind = field(action, G.action_kind);
    const int32_t context_subject = field(context, G.context_subject), context_kind = field(context, T.c_kind);
    const int32_t selection = field(action, G.action_selection);
    if (str_is(kind, "open_capability")) {
        const int32_t controls = F(offers, "capability_controls");
        return subject == context_subject && !none(controls) && contains(controls, field(action, G.action_panel));
    }
    if (str_is(context_kind, "capability")) {
        const int32_t panel = field(offers, G.offers_capability_panel);
        if (subject != context_subject) return false;
        if (str_is(kind, "close_capability")) return !V(F(panel, "mandatory")).number;
        for (int32_t choice : items_of(F(panel, "choices")))
            if (F(choice, "ref") == selection && F(choice, "kind") == kind)
                return !str_is(F(choice, "kind"), "trade_resource") || count_of(action) <= num(F(choice, "max_count"));
        return false;
    }
    if (str_is(context_kind, "dialog"))
        return str_is(kind, "choose_dialog") && subject == context_subject && ref_in(F(field(offers, G.offers_dialog_panel), "choices"), selection);
    const int32_t armies = field(offers, G.offers_armies);
    const int32_t context_panel = field(context, G.context_panel);
    if (!none(armies)) {
        if (str_is(kind, "split_stack") || str_is(kind, "merge_stacks") || str_is(kind, "swap_stacks")) {
            const int64_t maximum = army_operation_maximum(armies, str(kind), selection, field(action, G.action_destination));
            return subject == context_subject && maximum > 0 && (!str_is(kind, "split_stack") || count_of(action) <= maximum);
        }
        if (str_is(kind, "open_stack")) {
            int32_t army;
            const int32_t slot = first_slot(armies, selection, &army);
            return subject == context_subject && slot >= 0 && num(field(slot, G.slot_count)) > 0;
        }
        if (str_is(kind, "swap_garrison")) {
            if (!(str_is(context_kind, "town") && str_is(context_panel, "overview") && subject == context_subject)) return false;
            for (int32_t army : items_of(armies)) if (role_in(army, "garrison_hero", "visiting_hero")) return true;
            return false;
        }
        if (str_is(context_kind, "stack")) {
            if (subject != context_subject) return false;
            if (str_is(kind, "back")) return true;
            const int32_t stack = field(offers, G.offers_stack_panel);
            if (str_is(context_panel, "overview")) return str_is(kind, "select_upgrade") && ref_in(F(stack, "offers"), selection);
            return str_is(kind, "upgrade_stack") && V(F(stack, "affordable")).number && selection == F(stack, "selected_upgrade_ref");
        }
    }
    const int32_t panel = field(offers, G.offers_panel);
    if (!none(panel)) {
        if (str_is(context_panel, "overview")) {
            if (str_is(kind, "transfer_stack")) {
                const int32_t exchange = F(panel, "armies");
                return subject == context_subject && !none(exchange)
                       && count_of(action) <= exchange_maximum(exchange, selection, field(action, G.action_destination));
            }
            if (str_is(kind, "close")) return true;
            if (!(str_is(kind, "open_panel") && subject == context_subject)) return false;
            const int32_t which = field(action, G.action_panel);
            for (int32_t control : items_of(F(panel, "controls")))
                if (F(control, "panel") == which && (str_is(which, "hall") || str_is(which, "tavern") || F(control, "ref") == selection)) return true;
            return false;
        }
        if (subject != context_subject) return false;
        if (str_is(kind, "back")) return true;
        if (str_is(context_panel, "hall")) return str_is(kind, "open_building") && ref_in(F(panel, "buildings"), selection);
        if (str_is(context_panel, "tavern")) {
            const int32_t tavern = F(panel, "tavern");
            if (str_is(kind, "select_hero")) return ref_in(F(tavern, "offers"), selection);
            return str_is(kind, "hire_hero") && selection == F(tavern, "selected_offer_ref") && str_is(F(tavern, "recruit_status"), "allowed");
        }
        if (str_is(context_panel, "recruitment")) {
            const int32_t recruitment = F(panel, "recruitment");
            if (str_is(kind, "select_recruit")) return ref_in(F(recruitment, "offers"), selection);
            return str_is(kind, "recruit") && selection == F(recruitment, "selected_offer_ref") && count_of(action) <= num(F(recruitment, "slider_max"))
                   && field(action, G.action_destination) == F(F(recruitment, "destination"), "ref");
        }
        const int32_t building = F(panel, "building");
        return str_is(kind, "construct") && selection == F(building, "ref") && str_is(F(building, "status"), "allowed");
    }
    if (!str_is(context_kind, "map")) return str_is(kind, "close");
    if (str_is(kind, "end_turn")) return true;
    if ((str_is(kind, "open_hero") || str_is(kind, "move_to")) && contains(F(offers, "hero_refs"), subject)) {
        if (str_is(kind, "open_hero")) return true;
        const int32_t movable = F(offers, "movable_hero_refs");
        const auto &size = items_of(F(offers, "size"));
        return (none(movable) || contains(movable, subject)) && num(field(action, G.action_x)) < num(size[0]) && num(field(action, G.action_y)) < num(size[1]);
    }
    return str_is(kind, "open_town") && contains(field(offers, G.offers_town_refs), subject);
}

struct Revealed { std::vector<int64_t> row_major; std::vector<std::pair<int64_t, int64_t>> cells; };

const Revealed &revealed_cells(int32_t tiles, bool has_level, int64_t level, int64_t width) {  // guard._revealed and _row_major
    static std::map<std::tuple<int32_t, bool, int64_t, int64_t>, Revealed> known;
    auto key = std::make_tuple(tiles, has_level, level, width);
    auto found = known.find(key);
    if (found != known.end()) return found->second;
    if (known.size() >= 256) known.clear();
    Revealed made;
    if (has_level)
        for (int32_t tile : items_of(tiles)) {
            const auto &position = items_of(field(tile, T.tile_position));
            if (num(position[2]) == level) made.cells.push_back({num(position[0]), num(position[1])});
        }
    std::sort(made.cells.begin(), made.cells.end());
    made.cells.erase(std::unique(made.cells.begin(), made.cells.end()), made.cells.end());
    for (auto &cell : made.cells) made.row_major.push_back(cell.second * width + cell.first);
    std::sort(made.row_major.begin(), made.row_major.end());
    return known.emplace(key, std::move(made)).first->second;
}

struct WalkStep {
    std::string name;
    int kind = 0;  // 0 categorical, 1 quantity, 2 coordinate
    std::vector<int32_t> roots;
    std::vector<uint8_t> flags;  // window memory bits, empty when none is set
    int32_t root = -1;
    float numeric[4] = {0, 0, 0, 0}, offsets[2] = {0, 0};
    int64_t hx = 0, hy = 0, width = 0, height = 0;
    bool has_indices = false;
    std::vector<int64_t> indices;
    int selected = -1;
    size_t size() const { return kind == 0 ? roots.size() : kind == 1 ? 2 : has_indices ? indices.size() : (size_t)(width * height); }
};

struct Chooser {
    virtual ~Chooser() {}
    virtual int choose(WalkStep &step, int hint) = 0;  // the selected index; ``hint`` is the evaluated action's, or -1
};

const char *ARRANGEMENT_KINDS[] = {"split_stack", "merge_stacks", "swap_stacks"};
const char *ACTION_ORDER[] = {"move_to", "open_hero", "open_town", "end_turn", "close", "open_panel", "open_building", "construct", "back",
                              "select_recruit", "recruit", "transfer_stack", "select_hero", "hire_hero", "split_stack", "merge_stacks",
                              "swap_stacks", "open_stack", "select_upgrade", "upgrade_stack", "swap_garrison", "choose_dialog"};
const char *CAPABILITY_ACTION_LIST[] = {"trade_resource", "buy_artifact", "move_artifact", "assemble_artifact", "disassemble_artifact",
                                        "cast_adventure_spell", "choose_level_up", "choose_spell_target", "cancel_spell_target", "inspect_object"};

template <class Seq> bool sorted_has(const Seq &items, const typename Seq::value_type &item) { return std::binary_search(items.begin(), items.end(), item); }

struct Walk {
    int32_t observation, offers;
    const Support *guard;  // nullptr: no guard
    const std::set<Key> *memory;
    Chooser &chooser;
    std::vector<int32_t> trees;  // the trees the walk asked features of, in order
    std::unordered_map<int32_t, bool> permits_memo;
    int32_t context, context_subject, schema, v_count1, v_count2;
    std::unordered_map<std::string, int32_t> kind_values;

    Walk(int32_t observation, int32_t offers, const Support *guard, const std::set<Key> *memory, Chooser &chooser)
        : observation(observation), offers(offers), guard(guard), memory(memory), chooser(chooser) {
        prepare_actions();
        context = field(offers, G.offers_context);
        context_subject = field(context, G.context_subject);
        schema = F(offers, "schema");
        v_count1 = int_value(1); v_count2 = int_value(2);
    }

    int32_t kind_value(const std::string &kind) {
        auto found = kind_values.find(kind);
        if (found != kind_values.end()) return found->second;
        return kind_values[kind] = str_value(kind);
    }

    int32_t action(const std::string &kind, int32_t subject = -2, int32_t x = -1, int32_t y = -1, int32_t panel = -1, int32_t selection = -1,
                   int32_t count = -1, int32_t destination = -1) {  // _action: subject defaults to the context's
        if (subject == -2) subject = (kind == "close" || kind == "end_turn") ? v_none : context_subject;
        auto value = [](int32_t item) { return item < 0 ? v_none : item; };
        return make_action(ActionSpec{kind_value(kind), subject, value(x), value(y), value(panel), value(selection), value(count), value(destination)}, schema);
    }

    int32_t features(int32_t tree) { trees.push_back(tree); return tree; }

    // -- the guard (GuardSupport) ---------------------------------------------------------------------
    std::vector<int64_t> coordinates(int32_t subject, size_t limit) {  // GuardSupport.movement_coordinates
        const auto &size = items_of(F(offers, "size"));
        const int64_t width = num(size[0]);
        const Revealed &revealed = revealed_of(subject, width);
        std::vector<int64_t> out;
        for (int64_t index : revealed.row_major) {
            const int64_t x = index % width, y = index / width;
            const int32_t move = action("move_to", subject, int_value(x), int_value(y));
            if (offers_permit(offers, move) && permits_movement(subject, x, y, revealed)) {
                out.push_back(index);
                if (out.size() >= limit) break;
            }
        }
        return out;
    }

    const Revealed &revealed_of(int32_t subject, int64_t width) {
        bool has_level = false;
        int64_t level = 0;
        for (int32_t hero : items_of(field(observation, T.o_heroes)))
            if (field(hero, T.h_handle) == subject) { has_level = true; level = num(items_of(field(hero, G.hero_position))[2]); break; }
        return revealed_cells(field(observation, T.o_tiles), has_level, level, width);
    }

    bool permits_movement(int32_t subject, int64_t x, int64_t y, const Revealed &revealed) {
        const std::string &handle = str(subject);
        return sorted_has(guard->movable, handle) && !sorted_has(guard->blocked, Coordinate{handle, x, y})
               && std::binary_search(revealed.cells.begin(), revealed.cells.end(), std::make_pair(x, y));
    }

    bool repeats_command(int32_t candidate) {
        if (!guard->has_previous) return false;
        const std::string &kind = str(field(candidate, G.action_kind));
        const size_t cut = guard->previous.find(SEP);
        if (guard->previous.compare(0, cut, kind) != 0 || cut != kind.size()) return false;
        return guard->previous == command_key(candidate, offers);
    }

    bool map_has_alternative() {
        for (const char *kind : {"open_hero", "open_town"}) {
            const int32_t refs = std::string(kind) == "open_hero" ? F(offers, "hero_refs") : field(offers, G.offers_town_refs);
            for (int32_t subject : items_of(refs))
                if (guard_permits(make_action(ActionSpec{kind_value(kind), subject, v_none, v_none, v_none, v_none, v_none, v_none}, schema))) return true;
        }
        for (const std::string &subject : guard->movable)
            if (!coordinates(str_value(subject), 1).empty()) return true;
        return false;
    }

    bool guard_permits(int32_t candidate) {  // GuardSupport.permits
        const int32_t kind = field(candidate, G.action_kind);
        const int32_t armies = field(offers, G.offers_armies);
        const bool forced = mandatory(offers);
        const int32_t selection = field(candidate, G.action_selection), destination = field(candidate, G.action_destination);
        if (in_list(kind, ARRANGEMENT_KINDS, 3) && !guard->management.empty() && truthy_tuple(armies) && !forced) {
            std::set<std::string> subjects;
            int32_t army;
            if (first_slot(armies, selection, &army) >= 0) subjects.insert(str(field(army, G.army_subject)));
            if (first_slot(armies, destination, &army) >= 0) subjects.insert(str(field(army, G.army_subject)));
            if (subjects.empty()) subjects.insert(none(field(candidate, G.action_subject)) ? NONE_ELEMENT : str(field(candidate, G.action_subject)));
            for (const std::string &subject : subjects) if (sorted_has(guard->management, subject)) return false;
        }
        bool has_swap_key = false;
        SwapKey swap_key;
        if (str_is(kind, "swap_stacks") && !guard->swaps.empty() && truthy_tuple(armies) && !forced) {
            const auto places = slot_places(armies);
            auto left = places.find(selection), right = places.find(destination);
            if (left != places.end() && right != places.end()) {
                swap_key = swap_key_of(left->second, right->second);
                has_swap_key = true;
                if (sorted_has(guard->swaps, swap_key)) return false;
            }
        }
        if (!offers_permit(offers, candidate)) return false;
        if (forced || in_list(kind, CLOSE_KINDS, 3)) return true;
        if (guard->exhausted) return str_is(kind, "end_turn");
        if (repeats_command(candidate)) return str_is(kind, "end_turn") && !map_has_alternative();
        if (str_is(kind, "end_turn")) return true;
        if (str_is(kind, "move_to")) {
            const int32_t subject = field(candidate, G.action_subject);
            const int64_t width = num(items_of(F(offers, "size"))[0]);
            return permits_movement(subject, num(field(candidate, G.action_x)), num(field(candidate, G.action_y)), revealed_of(subject, width));
        }
        if (str_is(kind, "swap_stacks") && !guard->swaps.empty()) {
            if (!has_swap_key) {
                const auto places = slot_places(armies);
                swap_key = swap_key_of(places.at(selection), places.at(destination));
            }
            if (sorted_has(guard->swaps, swap_key)) return false;
        }
        if (management_kind(kind)) {
            if (guard->management.empty()) return true;
            for (const std::string &subject : management_subjects(candidate, offers)) if (sorted_has(guard->management, subject)) return false;
            return true;
        }
        Key key;
        if (!nav_key(candidate, offers, key)) return true;  // None is never blocked
        if (sorted_has(guard->navigation, key)) return false;
        return !sorted_has(guard->windows, split(key)[1]);
    }

    bool permitted(int32_t candidate) {
        if (guard == nullptr) return true;
        auto found = permits_memo.find(candidate);
        if (found != permits_memo.end()) return found->second;
        const bool result = guard_permits(candidate);
        permits_memo.emplace(candidate, result);
        return result;
    }

    bool repeated_quantity(int32_t candidate, int64_t &out) {  // GuardSupport.repeated_quantity; false for None
        if (guard == nullptr || !guard->has_previous || mandatory(offers)) return false;
        std::vector<std::string> previous = split(guard->previous);
        if (previous[0] != str(field(candidate, G.action_kind))) return false;
        std::vector<std::string> now = split(command_key(candidate, offers));
        if (previous.size() != now.size()) return false;
        for (size_t i = 0; i + 1 < now.size(); i++) if (previous[i] != now[i]) return false;
        const std::string &text = previous.back();
        char *end = nullptr;
        if (text.empty()) { PyErr_SetString(PyExc_ValueError, "invalid literal for int() with base 10: ''"); throw Fail(); }
        out = std::strtoll(text.c_str(), &end, 10);
        return true;
    }

    int32_t quantity_representative(const std::string &kind, int64_t maximum, int32_t selection, int32_t destination = -1) {
        const int32_t first = action(kind, -2, -1, -1, -1, selection, v_count1, destination);
        int64_t repeated;
        if (maximum > 1 && repeated_quantity(first, repeated) && repeated == 1) return action(kind, -2, -1, -1, -1, selection, v_count2, destination);
        return first;
    }

    int32_t pair_action(const std::string &kind, const Pair &pair) {
        if (kind == "transfer_stack" || kind == "split_stack") return quantity_representative(kind, pair.maximum, pair.source, pair.destination);
        return action(kind, -2, -1, -1, -1, pair.source, -1, pair.destination);
    }

    int32_t panel_action(int32_t control) {
        const int32_t panel = F(control, "panel");
        return action("open_panel", -2, -1, -1, panel, str_is(panel, "recruitment") ? F(control, "ref") : -1);
    }

    // -- kinds and representatives ----------------------------------------------------------------------
    std::vector<std::string> window_kinds() {
        const int32_t context_kind = field(context, T.c_kind), context_panel = field(context, G.context_panel);
        const int32_t armies = field(offers, G.offers_armies), panel = field(offers, G.offers_panel);
        if (str_is(context_kind, "dialog")) return {"choose_dialog"};
        if (str_is(context_kind, "map")) {
            const int32_t heroes = F(offers, "hero_refs"), towns = field(offers, G.offers_town_refs), listed = F(offers, "movable_hero_refs");
            const int32_t movable = none(listed) ? heroes : listed;
            std::vector<std::string> out;
            if (truthy_tuple(movable)) out.push_back("move_to");
            if (truthy_tuple(heroes)) out.push_back("open_hero");
            if (truthy_tuple(towns)) out.push_back("open_town");
            out.push_back("end_turn");
            return out;
        }
        if (str_is(context_kind, "stack")) {
            const int32_t stack = field(offers, G.offers_stack_panel);
            if (str_is(context_panel, "overview")) return truthy_tuple(F(stack, "offers")) ? std::vector<std::string>{"back", "select_upgrade"} : std::vector<std::string>{"back"};
            return V(F(stack, "affordable")).number ? std::vector<std::string>{"back", "upgrade_stack"} : std::vector<std::string>{"back"};
        }
        if (truthy_tuple(armies)) {
            std::vector<std::string> kinds = {"close"};
            if (!none(panel) && truthy_tuple(F(panel, "controls"))) kinds.push_back("open_panel");
            for (const char *kind : ARRANGEMENT_KINDS) if (!operation_choices(armies, kind).empty()) kinds.push_back(kind);
            bool stacked = false;
            for (int32_t army : items_of(armies)) for (int32_t slot : items_of(field(army, G.army_slots))) stacked = stacked || num(field(slot, G.slot_count));
            if (stacked) kinds.push_back("open_stack");
            if (str_is(context_kind, "town")) {
                bool garrison = false;
                for (int32_t army : items_of(armies)) garrison = garrison || role_in(army, "garrison_hero", "visiting_hero");
                if (garrison) kinds.push_back("swap_garrison");
            }
            return kinds;
        }
        if (none(panel)) return {"close"};
        if (str_is(context_panel, "overview")) {
            std::vector<std::string> kinds = truthy_tuple(F(panel, "controls")) ? std::vector<std::string>{"close", "open_panel"} : std::vector<std::string>{"close"};
            const int32_t exchange = F(panel, "armies");
            if (!none(exchange) && !exchange_choices(exchange).empty()) kinds.push_back("transfer_stack");
            return kinds;
        }
        if (str_is(context_panel, "hall")) return truthy_tuple(F(panel, "buildings")) ? std::vector<std::string>{"open_building", "back"} : std::vector<std::string>{"back"};
        if (str_is(context_panel, "building"))
            return str_is(F(F(panel, "building"), "status"), "allowed") ? std::vector<std::string>{"construct", "back"} : std::vector<std::string>{"back"};
        if (str_is(context_panel, "tavern")) {
            const int32_t tavern = F(panel, "tavern");
            std::vector<std::string> kinds = {"back"};
            if (truthy_tuple(F(tavern, "offers"))) kinds.push_back("select_hero");
            if (str_is(F(tavern, "recruit_status"), "allowed")) kinds.push_back("hire_hero");
            return kinds;
        }
        const int32_t recruitment = F(panel, "recruitment");
        return num(F(recruitment, "slider_max")) ? std::vector<std::string>{"back", "select_recruit", "recruit"} : std::vector<std::string>{"back", "select_recruit"};
    }

    std::vector<std::string> native_kinds() {
        const int32_t capability = field(offers, G.offers_capability_panel);
        if (!none(capability)) {
            std::vector<std::string> kinds;
            if (!V(F(capability, "mandatory")).number) kinds.push_back("close_capability");
            for (const char *kind : CAPABILITY_ACTION_LIST) {
                bool found = false;
                for (int32_t choice : items_of(F(capability, "choices"))) found = found || str_is(F(choice, "kind"), kind);
                if (found) kinds.push_back(kind);
            }
            return kinds;
        }
        std::vector<std::string> kinds = window_kinds();
        if (truthy_tuple(F(offers, "capability_controls"))) kinds.push_back("open_capability");
        return kinds;
    }

    std::vector<int32_t> subjects(const std::string &kind) {  // _subjects
        const int32_t listed = F(offers, "movable_hero_refs");
        const int32_t refs = kind == "open_town" ? field(offers, G.offers_town_refs) : (kind == "move_to" && !none(listed)) ? listed : F(offers, "hero_refs");
        std::vector<int32_t> out;
        for (int32_t ref : items_of(refs)) {
            if (guard == nullptr) out.push_back(ref);
            else if (kind == "move_to") { if (sorted_has(guard->movable, str(ref)) && !coordinates(ref, 1).empty()) out.push_back(ref); }
            else if (permitted(action(kind, ref))) out.push_back(ref);
        }
        return out;
    }

    // Each representative of ``kind`` in order until ``visit`` returns true (_representatives)
    template <class Visit> void representatives(const std::string &kind, Visit visit) {
        if (kind == "move_to" || kind == "open_hero" || kind == "open_town") {
            const int64_t width = num(items_of(F(offers, "size"))[0]);
            for (int32_t ref : subjects(kind)) {
                if (kind == "move_to") {
                    std::vector<int64_t> first = coordinates(ref, 1);
                    if (!first.empty() && visit(action(kind, ref, int_value(first[0] % width), int_value(first[0] / width)))) return;
                } else if (visit(action(kind, ref))) return;
            }
        } else if (kind == "open_capability") {
            for (int32_t panel : items_of(F(offers, "capability_controls"))) if (visit(action(kind, -2, -1, -1, panel))) return;
        } else if (in_list_text(kind, CAPABILITY_ACTION_LIST, 10)) {
            for (int32_t choice : items_of(F(field(offers, G.offers_capability_panel), "choices")))
                if (str_is(F(choice, "kind"), kind.c_str())) {
                    const int32_t made = kind == "trade_resource" ? quantity_representative(kind, num(F(choice, "max_count")), F(choice, "ref"))
                                                                  : action(kind, -2, -1, -1, -1, F(choice, "ref"));
                    if (visit(made)) return;
                }
        } else if (kind == "choose_dialog") {
            for (int32_t choice : items_of(F(field(offers, G.offers_dialog_panel), "choices"))) if (visit(action(kind, -2, -1, -1, -1, F(choice, "ref")))) return;
        } else if (kind == "open_panel") {
            for (int32_t control : items_of(F(field(offers, G.offers_panel), "controls"))) if (visit(panel_action(control))) return;
        } else if (kind == "open_stack") {
            for (int32_t army : items_of(field(offers, G.offers_armies)))
                for (int32_t slot : items_of(field(army, G.army_slots)))
                    if (num(field(slot, G.slot_count)) && visit(action(kind, -2, -1, -1, -1, field(slot, G.slot_ref)))) return;
        } else if (kind == "select_upgrade" || kind == "open_building" || kind == "select_recruit" || kind == "select_hero") {
            for (int32_t choice : items_of(selection_choices(kind))) if (visit(action(kind, -2, -1, -1, -1, F(choice, "ref")))) return;
        } else if (kind == "transfer_stack" || in_list_text(kind, ARRANGEMENT_KINDS, 3)) {
            const std::vector<Pair> choices = kind == "transfer_stack" ? exchange_choices(F(field(offers, G.offers_panel), "armies"))
                                                                       : operation_choices(field(offers, G.offers_armies), kind);
            for (const Pair &pair : choices) if (visit(pair_action(kind, pair))) return;
        } else if (kind == "recruit") {
            const int32_t recruit = F(field(offers, G.offers_panel), "recruitment");
            visit(quantity_representative(kind, num(F(recruit, "slider_max")), F(recruit, "selected_offer_ref"), F(F(recruit, "destination"), "ref")));
        } else if (kind == "construct") {
            visit(action(kind, -2, -1, -1, -1, F(F(field(offers, G.offers_panel), "building"), "ref")));
        } else if (kind == "hire_hero") {
            visit(action(kind, -2, -1, -1, -1, F(F(field(offers, G.offers_panel), "tavern"), "selected_offer_ref")));
        } else if (kind == "upgrade_stack") {
            visit(action(kind, -2, -1, -1, -1, F(field(offers, G.offers_stack_panel), "selected_upgrade_ref")));
        } else {
            visit(action(kind));
        }
    }

    static bool in_list_text(const std::string &kind, const char *const *list, size_t count) {
        for (size_t i = 0; i < count; i++) if (kind == list[i]) return true;
        return false;
    }

    int32_t selection_choices(const std::string &kind) {
        if (kind == "select_upgrade") return F(field(offers, G.offers_stack_panel), "offers");
        const int32_t panel = field(offers, G.offers_panel);
        if (kind == "open_building") return F(panel, "buildings");
        if (kind == "select_hero") return F(F(panel, "tavern"), "offers");
        return F(F(panel, "recruitment"), "offers");
    }

    std::vector<std::string> kinds() {  // _kinds
        std::vector<std::string> all = native_kinds();
        if (guard == nullptr) return all;
        const int32_t armies = field(offers, G.offers_armies);
        bool barred = !guard->management.empty() && truthy_tuple(armies) && !mandatory(offers);
        if (barred)
            for (int32_t army : items_of(armies)) if (!sorted_has(guard->management, str(field(army, G.army_subject)))) { barred = false; break; }
        std::vector<std::string> out;
        for (const std::string &kind : all) {
            if (barred && in_list_text(kind, ARRANGEMENT_KINDS, 3)) continue;
            bool any = false;
            representatives(kind, [&](int32_t candidate) { return any = permitted(candidate); });
            if (any) out.push_back(kind);
        }
        return out;
    }

    std::map<std::string, std::vector<uint8_t>> window_flags(const std::vector<std::string> &kinds) {  // _window_flags
        std::map<std::string, std::vector<uint8_t>> flags;
        std::set<std::string> remembered;
        for (const Key &key : *memory) remembered.insert(split(key)[0]);
        for (const std::string &kind : kinds) {
            if (!remembered.count(kind)) continue;
            std::vector<uint8_t> bits;
            representatives(kind, [&](int32_t candidate) {
                if (permitted(candidate)) {
                    Key key;
                    const bool has = nav_key(candidate, offers, key);
                    bits.push_back(has && memory->count(key) ? 1 : 0);
                }
                return false;
            });
            flags[kind] = bits;
        }
        return flags;
    }

    // -- choices ---------------------------------------------------------------------------------------
    int categorical(const char *name, std::vector<int32_t> semantics, const std::vector<uint8_t> &flags, int hint) {
        WalkStep step;
        step.name = name;
        for (int32_t tree : semantics) step.roots.push_back(features(tree));
        bool any = false;
        for (uint8_t bit : flags) any = any || bit;
        if (any) step.flags = flags;
        return chooser.choose(step, hint);
    }

    int64_t choose_count(int32_t feature, int64_t maximum, int64_t selected, bool has_excluded, int64_t excluded) {  // _choose_count
        int64_t low = 1, high = maximum, depth = 0;
        while (low < high) {
            const int64_t middle = floor_div(low + high, 2);
            int branch;
            if (has_excluded && excluded == low && low == middle) branch = 1;
            else if (has_excluded && excluded == high && high == middle + 1) branch = 0;
            else {
                WalkStep step;
                step.name = "count:" + std::to_string(low) + ":" + std::to_string(high);
                step.kind = 1;
                step.root = feature;
                step.numeric[0] = (float)((double)low / (double)maximum); step.numeric[1] = (float)((double)high / (double)maximum);
                step.numeric[2] = (float)std::log1p((double)maximum); step.numeric[3] = (float)((double)depth / 31.0);
                const int64_t centre = floor_div(low + high, 2);
                step.offsets[0] = (float)std::log1p((double)(centre - low)); step.offsets[1] = (float)std::log1p((double)(high - centre - 1));
                branch = chooser.choose(step, selected > 0 ? (int)(selected > middle) : -1);
            }
            if (branch) low = middle + 1; else high = middle;
            depth++;
        }
        return low;
    }

    static int64_t floor_div(int64_t a, int64_t b) { int64_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return q; }

    int index_of(const std::vector<int32_t> &items, int32_t item) {
        for (size_t i = 0; i < items.size(); i++) if (items[i] == item) return (int)i;
        PyErr_SetString(PyExc_ValueError, "the evaluated action is not among the choices"); throw Fail();
    }

    int32_t run(int32_t given) {  // _walk: the action (sampled, or ``given`` evaluated)
        if (given >= 0 && (!offers_permit(offers, given) || !permitted(given))) {
            PyErr_SetString(PyExc_ValueError, "Action is outside this exact behavior support"); throw Fail();
        }
        auto hint = [&](int index) { return given >= 0 ? index : -1; };
        const std::vector<std::string> all = kinds();
        const auto flags = (memory != nullptr && !memory->empty()) ? window_flags(all) : std::map<std::string, std::vector<uint8_t>>();
        std::set<std::string> fully;
        for (auto &entry : flags) {
            bool every = !entry.second.empty();
            for (uint8_t bit : entry.second) every = every && bit;
            if (every) fully.insert(entry.first);
        }
        std::vector<uint8_t> stale;
        if (!fully.empty()) for (const std::string &kind : all) stale.push_back(fully.count(kind) ? 1 : 0);
        std::vector<int32_t> kind_trees;
        const int32_t kind_tag = text_node("action_kind");
        for (const std::string &kind : all) kind_trees.push_back(node(kind_tag, {text_node(kind.c_str())}));
        int given_kind = -1;
        if (given >= 0) {
            const std::string &wanted = str(field(given, G.action_kind));
            for (size_t i = 0; i < all.size(); i++) if (all[i] == wanted) given_kind = (int)i;
            if (given_kind < 0) { PyErr_SetString(PyExc_ValueError, "kind is not in list"); throw Fail(); }
        }
        const std::string kind = all.at((size_t)categorical("kind", kind_trees, stale, given_kind));
        auto found_flags = flags.find(kind);
        const std::vector<uint8_t> no_flags, &target_flags = found_flags == flags.end() ? no_flags : found_flags->second;
        int32_t subject = v_none, panel = v_none, selection = v_none, destination = v_none, x = v_none, y = v_none, count = v_none;
        auto given_field = [&](int32_t index) { return given >= 0 ? field(given, index) : -1; };
        if (kind == "open_capability") {
            std::vector<int32_t> controls, semantics;
            for (int32_t item : items_of(F(offers, "capability_controls"))) if (permitted(action(kind, -2, -1, -1, item))) controls.push_back(item);
            for (int32_t item : controls) semantics.push_back(node("capability_panel", {vtext(item)}));
            const int index = categorical("capability_panel", semantics, target_flags, given >= 0 ? index_of(controls, given_field(G.action_panel)) : -1);
            subject = context_subject; panel = controls.at((size_t)index);
        } else if (kind == "close_capability") {
            subject = context_subject;
        } else if (in_list_text(kind, CAPABILITY_ACTION_LIST, 10)) {
            std::vector<int32_t> choices, semantics, refs;
            for (int32_t item : items_of(F(field(offers, G.offers_capability_panel), "choices"))) {
                if (!str_is(F(item, "kind"), kind.c_str())) continue;
                const int32_t made = kind == "trade_resource" ? quantity_representative(kind, num(F(item, "max_count")), F(item, "ref"))
                                                              : action(kind, -2, -1, -1, -1, F(item, "ref"));
                if (permitted(made)) { choices.push_back(item); refs.push_back(F(item, "ref")); }
            }
            for (int32_t item : choices) semantics.push_back(vcapability_choice(item));
            const int index = categorical("capability_choice", semantics, {}, given >= 0 ? index_of(refs, given_field(G.action_selection)) : -1);
            const int32_t choice = choices.at((size_t)index);
            subject = context_subject; selection = F(choice, "ref");
            if (kind == "trade_resource") {
                const int32_t feature = features(vcapability_choice(choice));
                int64_t excluded = 0;
                const bool has = repeated_quantity(action(kind, -2, -1, -1, -1, selection, v_count1), excluded);
                count = int_value(choose_count(feature, num(F(choice, "max_count")), given >= 0 ? count_of(given) : 0, has, excluded));
            }
        } else if (kind == "choose_dialog") {
            const int32_t delivered = field(offers, G.offers_dialog_panel);
            std::vector<int32_t> semantics, refs;
            for (int32_t item : items_of(F(delivered, "choices"))) { semantics.push_back(vdialog_choice(delivered, item)); refs.push_back(F(item, "ref")); }
            const int index = categorical("dialog_choice", semantics, {}, given >= 0 ? index_of(refs, given_field(G.action_selection)) : -1);
            subject = context_subject; selection = refs.at((size_t)index);
        } else if (kind == "move_to" || kind == "open_hero" || kind == "open_town") {
            const std::vector<int32_t> refs = subjects(kind);
            std::vector<int32_t> semantics;
            const int32_t tag = text_node(kind.c_str());
            for (int32_t ref : refs) semantics.push_back(node(tag, {vsubject(observation, ref)}));
            int index = categorical("subject", semantics, target_flags, given >= 0 ? index_of(refs, given_field(G.action_subject)) : -1);
            subject = refs.at((size_t)index);
            if (kind == "move_to") {
                const auto &size = items_of(F(offers, "size"));
                const int64_t width = num(size[0]), height = num(size[1]);
                WalkStep step;
                step.name = "coordinate"; step.kind = 2; step.width = width; step.height = height;
                step.root = features(semantics[(size_t)index]);
                if (guard != nullptr) {
                    bool found = false;
                    for (int32_t hero : items_of(field(observation, T.o_heroes)))
                        if (field(hero, T.h_handle) == subject) {
                            const auto &position = items_of(field(hero, G.hero_position));
                            step.hx = num(position[0]); step.hy = num(position[1]); found = true; break;
                        }
                    if (!found) { PyErr_SetString(PyExc_StopIteration, "the moving hero"); throw Fail(); }
                }
                const int64_t wanted = given >= 0 ? num(field(given, G.action_y)) * width + num(field(given, G.action_x)) : -1;
                int64_t cell;
                if (guard == nullptr) {
                    cell = chooser.choose(step, (int)wanted);
                } else {
                    step.has_indices = true;
                    step.indices = coordinates(subject, (size_t)-1);
                    int position = -1;
                    if (given >= 0) {
                        for (size_t i = 0; i < step.indices.size(); i++) if (step.indices[i] == wanted) position = (int)i;
                        if (position < 0) { PyErr_SetString(PyExc_ValueError, "coordinate is not in tuple"); throw Fail(); }
                    }
                    const std::vector<int64_t> indices = step.indices;
                    cell = indices.at((size_t)chooser.choose(step, position));
                }
                x = int_value(cell % width); y = int_value(cell / width);
            }
        } else if (kind == "open_panel") {
            const int32_t delivered = field(offers, G.offers_panel);
            std::vector<int32_t> all_controls = items_of(F(delivered, "controls"));
            for (const char *which : {"hall", "tavern"}) {
                int copies = 0;
                for (int32_t control : all_controls) copies += str_is(F(control, "panel"), which);
                if (copies > 1) { PyErr_SetString(PyExc_ValueError, "Duplicate semantic town panel controls"); throw Fail(); }
            }
            std::vector<int32_t> controls, semantics;
            for (int32_t control : all_controls) if (permitted(panel_action(control))) controls.push_back(control);
            int selected = -1;
            if (given >= 0)
                for (size_t i = 0; i < controls.size(); i++) {
                    const int32_t which = F(controls[i], "panel");
                    if (which == field(given, G.action_panel) && (str_is(which, "hall") || str_is(which, "tavern") || F(controls[i], "ref") == field(given, G.action_selection))) {
                        selected = (int)i; break;
                    }
                }
            for (int32_t control : controls) semantics.push_back(vsemantic(control));
            const int index = categorical("control", semantics, target_flags, selected);
            const int32_t control = controls.at((size_t)index);
            subject = context_subject; panel = F(control, "panel");
            selection = str_is(panel, "recruitment") ? F(control, "ref") : v_none;
        } else if (kind == "open_stack") {
            std::vector<int32_t> choices, semantics;
            for (int32_t army : items_of(field(offers, G.offers_armies)))
                for (int32_t slot : items_of(field(army, G.army_slots)))
                    if (num(field(slot, G.slot_count)) && permitted(action(kind, -2, -1, -1, -1, field(slot, G.slot_ref)))) choices.push_back(field(slot, G.slot_ref));
            for (int32_t ref : choices) semantics.push_back(varmy_slot(observation, ref, false));
            const int index = categorical("selection", semantics, target_flags, given >= 0 ? index_of(choices, given_field(G.action_selection)) : -1);
            subject = context_subject; selection = choices.at((size_t)index);
        } else if (kind == "select_upgrade" || kind == "open_building" || kind == "select_recruit" || kind == "select_hero") {
            std::vector<int32_t> choices, semantics, refs;
            for (int32_t item : items_of(selection_choices(kind)))
                if (permitted(action(kind, -2, -1, -1, -1, F(item, "ref")))) { choices.push_back(item); refs.push_back(F(item, "ref")); }
            for (int32_t item : choices) semantics.push_back(vsemantic(item));
            const int index = categorical("selection", semantics, kind == "select_upgrade" ? no_flags : target_flags,
                                          given >= 0 ? index_of(refs, given_field(G.action_selection)) : -1);
            subject = context_subject; selection = refs.at((size_t)index);
        } else if (kind == "transfer_stack" || in_list_text(kind, ARRANGEMENT_KINDS, 3)) {
            const int32_t exchange = kind == "transfer_stack" ? F(field(offers, G.offers_panel), "armies") : -1;
            const std::vector<Pair> listed = exchange >= 0 ? exchange_choices(exchange) : operation_choices(field(offers, G.offers_armies), kind);
            std::vector<Pair> choices;
            for (const Pair &pair : listed) if (permitted(pair_action(kind, pair))) choices.push_back(pair);
            std::vector<int32_t> sources;
            for (const Pair &pair : choices) if (std::find(sources.begin(), sources.end(), pair.source) == sources.end()) sources.push_back(pair.source);
            std::vector<int32_t> source_semantics;
            const int32_t kind_tag_node = text_node(kind.c_str());
            for (int32_t ref : sources) {
                const int32_t slot = varmy_slot(observation, ref, false);
                source_semantics.push_back(kind != "transfer_stack" ? node(kind_tag_node, {slot}) : slot);
            }
            const int source_index = categorical("source_slot", source_semantics, {}, given >= 0 ? index_of(sources, given_field(G.action_selection)) : -1);
            selection = sources.at((size_t)source_index);
            std::vector<int32_t> destinations, destination_semantics;
            for (const Pair &pair : choices) if (pair.source == selection) destinations.push_back(pair.destination);
            const int32_t pair_tag = text_node(kind == "transfer_stack" ? "transfer_pair" : (kind + "_pair").c_str());
            for (int32_t ref : destinations)
                destination_semantics.push_back(node(pair_tag, {varmy_slot(observation, selection, false), varmy_slot(observation, ref, true)}));
            const int destination_index = categorical("destination_slot", destination_semantics, {},
                                                      given >= 0 ? index_of(destinations, given_field(G.action_destination)) : -1);
            destination = destinations.at((size_t)destination_index);
            subject = context_subject;
            const int64_t maximum = exchange >= 0 ? exchange_maximum(exchange, selection, destination)
                                                  : army_operation_maximum(field(offers, G.offers_armies), kind, selection, destination);
            const int32_t feature = features(node(kind == "transfer_stack" ? "transfer_quantity" : (kind + "_quantity").c_str(),
                                                  {varmy_slot(observation, selection, false), varmy_slot(observation, destination, true)}));
            if (kind == "transfer_stack" || kind == "split_stack") {
                int64_t excluded = 0;
                const bool has = repeated_quantity(action(kind, -2, -1, -1, -1, selection, v_count1, destination), excluded);
                count = int_value(choose_count(feature, maximum, given >= 0 ? count_of(given) : 0, has, excluded));
            }
        } else if (kind == "construct" || kind == "back" || kind == "recruit" || kind == "hire_hero" || kind == "swap_garrison" || kind == "upgrade_stack") {
            subject = context_subject;
            if (kind == "upgrade_stack") selection = F(field(offers, G.offers_stack_panel), "selected_upgrade_ref");
            else if (kind == "construct") selection = F(F(field(offers, G.offers_panel), "building"), "ref");
            else if (kind == "hire_hero") selection = F(F(field(offers, G.offers_panel), "tavern"), "selected_offer_ref");
            else if (kind == "recruit") {
                const int32_t recruit = F(field(offers, G.offers_panel), "recruitment");
                selection = F(recruit, "selected_offer_ref"); destination = F(F(recruit, "destination"), "ref");
                const int32_t chosen = find_by_ref(F(recruit, "offers"), selection);
                const int32_t feature = features(node("quantity", {vsemantic(chosen), vsemantic(F(recruit, "destination"))}));
                int64_t excluded = 0;
                const bool has = repeated_quantity(action(kind, -2, -1, -1, -1, selection, v_count1, destination), excluded);
                count = int_value(choose_count(feature, num(F(recruit, "slider_max")), given >= 0 ? count_of(given) : 0, has, excluded));
            }
        }
        const int32_t chosen = make_action(ActionSpec{kind_value(kind), subject, x, y, panel, selection, count, destination}, schema);
        if (!offers_permit(offers, chosen) || !permitted(chosen) || (given >= 0 && chosen != given)) {
            PyErr_SetString(PyExc_ValueError, "Conditional action construction disagrees with support"); throw Fail();
        }
        return chosen;
    }

    int32_t vcapability_choice(int32_t choice) {  // policy._capability_choice
        const int32_t destination = F(choice, "destination");
        return node("capability_choice", {vsemantic(choice), node("actor", {vsubject(observation, F(choice, "subject"))}),
                                          !none(destination) ? node("recipient", {vsubject(observation, destination)}) : node("no_recipient")});
    }
};

struct Replay : Chooser {  // evaluates a given action: each step takes the action's own index
    std::vector<WalkStep> steps;
    int choose(WalkStep &step, int hint) override {
        if (hint < 0 || (size_t)hint >= step.size()) { PyErr_SetString(PyExc_ValueError, "Recorded action is outside the compiled conditional support"); throw Fail(); }
        step.selected = hint;
        steps.push_back(step);
        return hint;
    }
};

PyObject *py_step(const WalkStep &step) {
    Ref roots(PyList_New(0));
    for (int32_t root : step.roots) if (PyList_Append(roots.get(), python_tree(root))) throw Fail();
    Ref flags(PyTuple_New((Py_ssize_t)step.flags.size()));
    for (size_t i = 0; i < step.flags.size(); i++) PyTuple_SET_ITEM(flags.get(), (Py_ssize_t)i, PyBool_FromLong(step.flags[i]));
    Ref indices(PyTuple_New((Py_ssize_t)step.indices.size()));
    for (size_t i = 0; i < step.indices.size(); i++) PyTuple_SET_ITEM(indices.get(), (Py_ssize_t)i, PyLong_FromLongLong(step.indices[i]));
    PyObject *root = step.root >= 0 ? python_tree(step.root) : Py_None;
    return Py_BuildValue("{s:s#,s:i,s:O,s:O,s:O,s:(ffff),s:(ff),s:(LLLL),s:O,s:i}", "name", step.name.data(), (Py_ssize_t)step.name.size(),
                         "kind", step.kind, "roots", roots.get(), "flags", flags.get(), "root", root,
                         "numeric", step.numeric[0], step.numeric[1], step.numeric[2], step.numeric[3], "offsets", step.offsets[0], step.offsets[1],
                         "coordinate", (long long)step.hx, (long long)step.hy, (long long)step.width, (long long)step.height,
                         "indices", step.has_indices ? indices.get() : Py_None, "selected", step.selected);
}

PyObject *py_walks(PyObject *, PyObject *args) {  // walks(producer, guarded): (action, trees, steps) at each answered decision (tests)
    PyObject *object;
    int guarded;
    if (!PyArg_ParseTuple(args, "Op", &object, &guarded)) return nullptr;
    if (Py_TYPE(object) != &ProducerType) { PyErr_SetString(PyExc_TypeError, "a Producer"); return nullptr; }
    Producer *producer = (Producer *)object;
    try {
        prepare_guard();
        const std::vector<NFrame> &frames = *producer->frames;
        Tracker tracker;
        WindowMemory memory;
        Ref out(PyList_New(0));
        Support support;
        std::set<Key> keys;
        for (size_t i = 0; i < frames.size(); i++) {
            const bool decided = guarded ? tracker.observe(frames[i], support) : frames[i].kind == F_DECISION;
            memory.observe(frames, (int32_t)i, keys);
            if (!decided) continue;
            int32_t given = -1;
            for (size_t j = i + 1; j < frames.size() && frames[j].kind != F_DECISION; j++)
                if (frames[j].kind == F_SUBMITTED && frames[j].decision_seq == frames[i].seq) { given = frames[j].action; break; }
            if (given < 0) continue;
            Replay replay;
            Walk walk(frames[i].observation, frames[i].offers, guarded ? &support : nullptr, &keys, replay);
            int32_t chosen;
            try {
                chosen = walk.run(given);
            } catch (Fail &) {  // the decision's error in place of its walk
                PyObject *type, *value, *trace;
                PyErr_Fetch(&type, &value, &trace);
                Ref entry(Py_BuildValue("(iOOO)", frames[i].seq, value ? value : Py_None, python_value(given), Py_None));
                Py_XDECREF(type); Py_XDECREF(value); Py_XDECREF(trace);
                if (PyList_Append(out.get(), entry.get())) throw Fail();
                continue;
            }
            Ref trees(PyList_New(0));
            for (int32_t tree : walk.trees) if (PyList_Append(trees.get(), python_tree(tree))) throw Fail();
            Ref steps(PyList_New(0));
            for (const WalkStep &step : replay.steps) { Ref item(py_step(step)); if (PyList_Append(steps.get(), item.get())) throw Fail(); }
            Ref entry(Py_BuildValue("(iOOO)", frames[i].seq, python_value(chosen), trees.get(), steps.get()));
            if (PyList_Append(out.get(), entry.get())) throw Fail();
        }
        Py_INCREF(out.get());
        return out.get();
    } catch (Fail &) {
        return nullptr;
    }
}

PyObject *py_walk_times(PyObject *, PyObject *object) {  // walk_times(producer): seconds in (tracker and memory, walks) (tests)
    if (Py_TYPE(object) != &ProducerType) { PyErr_SetString(PyExc_TypeError, "a Producer"); return nullptr; }
    Producer *producer = (Producer *)object;
    try {
        prepare_guard();
        const std::vector<NFrame> &frames = *producer->frames;
        Tracker tracker;
        WindowMemory memory;
        Support support;
        std::set<Key> keys;
        double follow = 0, walking = 0;
        long walks = 0;
        for (size_t i = 0; i < frames.size(); i++) {
            auto start = std::chrono::steady_clock::now();
            const bool decided = tracker.observe(frames[i], support);
            memory.observe(frames, (int32_t)i, keys);
            auto middle = std::chrono::steady_clock::now();
            follow += std::chrono::duration<double>(middle - start).count();
            if (!decided) continue;
            int32_t given = -1;
            for (size_t j = i + 1; j < frames.size() && frames[j].kind != F_DECISION; j++)
                if (frames[j].kind == F_SUBMITTED && frames[j].decision_seq == frames[i].seq) { given = frames[j].action; break; }
            if (given < 0) continue;
            middle = std::chrono::steady_clock::now();
            Replay replay;
            Walk walk(frames[i].observation, frames[i].offers, &support, &keys, replay);
            walk.run(given);
            walking += std::chrono::duration<double>(std::chrono::steady_clock::now() - middle).count();
            walks++;
        }
        return Py_BuildValue("(ddl)", follow, walking, walks);
    } catch (Fail &) {
        return nullptr;
    }
}
PyMethodDef methods[] = {
    {"configure", py_configure, METH_VARARGS, "configure(policy, contract, fn_text, fn_number, fn_node)"},
    {"observation_tree", py_observation_tree, METH_O, "policy.observation_semantics, from the native DAG"},
    {"semantic_tree", py_semantic_tree, METH_O, "policy._semantic, from the native DAG"},
    {"observation_slot", (PyCFunction)(void (*)(void))py_observation_slot, METH_FASTCALL,
     "observation_slot(net, epoch, observation): the arena slot of the observation's tree"},
    {"stats", py_stats, METH_NOARGS, "sizes of the native tables"},
    {"configure_values", py_configure_values, METH_VARARGS, "configure_values(types, specs, missing): the contract schema"},
    {"decode", py_decode, METH_VARARGS, "decode(bytes, type name): the value id of one JSON document"},
    {"python_value", py_python_value, METH_O, "python_value(id): the contract object a value stands for"},
    {"set_native_failure", py_set_native_failure, METH_O, "the exception an engine error message raises"},
    {"value_tree", py_value_tree, METH_VARARGS, "value_tree(value, observation=0): the semantic tree of a value"},
    {"payloads", py_payloads, METH_O, "payloads(producer): each frame's payload tree (tests)"},
    {"window_memories", py_window_memories, METH_O, "window_memories(producer): WindowMemory keys at each decision (tests)"},
    {"walk_times", py_walk_times, METH_O, "walk_times(producer): seconds following frames and walking (tests)"},
    {"walks", py_walks, METH_VARARGS, "walks(producer, guarded): the walk of each answered decision (tests)"},
    {"guard_supports", py_guard_supports, METH_VARARGS, "guard_supports(producer, management, windows, decisions): GuardSupport fields (tests)"},
    {nullptr, nullptr, 0, nullptr}};

PyModuleDef module = {PyModuleDef_HEAD_INIT, "heroes_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit_heroes_native(void) {
    ProducerType.tp_name = "heroes_native.Producer";
    ProducerType.tp_basicsize = sizeof(Producer);
    ProducerType.tp_flags = Py_TPFLAGS_DEFAULT;
    ProducerType.tp_new = PyType_GenericNew;
    ProducerType.tp_init = (initproc)Producer_init;
    ProducerType.tp_dealloc = (destructor)Producer_dealloc;
    ProducerType.tp_methods = Producer_methods;
    ProducerType.tp_getset = Producer_getset;
    if (PyType_Ready(&ProducerType) < 0) return nullptr;
    PyObject *made = PyModule_Create(&module);
    if (!made) return nullptr;
    Py_INCREF(&ProducerType);
    if (PyModule_AddObject(made, "Producer", (PyObject *)&ProducerType) < 0) return nullptr;
    return made;
}
