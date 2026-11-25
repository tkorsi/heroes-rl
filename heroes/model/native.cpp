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

const std::vector<int32_t> &world_of(int32_t observation) {  // guard._world, as value ids (equal worlds, equal ids)
    static std::unordered_map<int32_t, std::vector<int32_t>> known;
    auto found = known.find(observation);
    if (found != known.end()) return found->second;
    if (known.size() >= 65536) known.clear();
    std::vector<int32_t> world;
    world.push_back(field(observation, T.o_size));
    std::vector<int32_t> tiles = V(field(observation, T.o_tiles)).items;
    auto position = [](int32_t tile) {
        std::vector<int64_t> out;
        for (int32_t item : V(field(tile, T.tile_position)).items) out.push_back(num(item));
        return out;
    };
    std::stable_sort(tiles.begin(), tiles.end(), [&](int32_t a, int32_t b) { return position(a) < position(b); });
    world.push_back(-1); world.insert(world.end(), tiles.begin(), tiles.end());
    std::vector<int32_t> objects = V(field(observation, T.o_objects)).items;
    std::stable_sort(objects.begin(), objects.end(), [](int32_t a, int32_t b) {
        return str(vfield(a, V(a).type, "handle")) < str(vfield(b, V(b).type, "handle")); });
    world.push_back(-2); world.insert(world.end(), objects.begin(), objects.end());
    world.push_back(-3); world.push_back(vfield(observation, T.observation, "resources"));
    std::vector<int32_t> heroes = V(field(observation, T.o_heroes)).items;
    std::stable_sort(heroes.begin(), heroes.end(), [](int32_t a, int32_t b) {
        return str(field(a, T.h_handle)) < str(field(b, T.h_handle)); });
    world.push_back(-4);
    for (int32_t hero : heroes) { world.push_back(field(hero, T.h_handle)); world.push_back(field(hero, G.hero_position)); world.push_back(field(hero, G.hero_movement)); }
    std::vector<int32_t> towns = V(field(observation, T.o_towns)).items;
    std::stable_sort(towns.begin(), towns.end(), [](int32_t a, int32_t b) {
        return str(field(a, T.t_handle)) < str(field(b, T.t_handle)); });
    world.push_back(-5); world.insert(world.end(), towns.begin(), towns.end());
    return known.emplace(observation, std::move(world)).first->second;
}

struct Tracker {
    int64_t management_limit = 32, window_limit = -1, decision_limit = -1;
    std::map<std::string, int64_t> windows, management;
    std::map<Key, int64_t> navigation;
    int64_t decisions = 0;
    bool has_day = false;
    int32_t day = 0, last_seq = -1;
    const std::vector<int32_t> *world = nullptr;
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
            const std::vector<int32_t> &now = world_of(frame.observation);
            const bool managing = consumed && action >= 0 && management_kind(field(action, G.action_kind));
            if (world != nullptr && *world != now && !managing) progress();
            world = &now;
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
