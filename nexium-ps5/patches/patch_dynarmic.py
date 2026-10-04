import pathlib
import shutil
import sys

root = pathlib.Path(sys.argv[1])
inc_src = pathlib.Path(sys.argv[2])
dyn = root / "vendor/dynarmic"
LF = "\n"
CRLF = "\r\n"


def read(path):
    with open(path, newline="") as f:
        text = f.read()
    return text, (CRLF if CRLF in text else LF)


def write(path, text):
    with open(path, "w", newline="") as f:
        f.write(text)


def replace(path, old, new, count=1):
    text, nl = read(path)
    old, new = old.replace(LF, nl), new.replace(LF, nl)
    found = text.count(old)
    if found != count:
        raise SystemExit(f"{path}: expected {count} of {old!r}, found {found}")
    write(path, text.replace(old, new))


def insert_before(path, marker, block):
    text, nl = read(path)
    idx = text.find(marker)
    if idx < 0:
        raise SystemExit(f"{path}: marker {marker!r} missing")
    write(path, text[:idx] + block.replace(LF, nl) + text[idx:])


shutil.copyfile(inc_src, dyn / "ps5_page_table.inc")

cpp = dyn / "dynarmic.cpp"
insert_before(cpp, "FQL khash_t(memory) *dynarmic_init_memory() {", '#include "ps5_page_table.inc"\n\n')
replace(cpp, "dynarmic->page_table[idx] = nullptr;", "page_table_store(dynarmic->page_table, idx, nullptr);")
replace(cpp, "{ dynarmic->page_table[idx] = (void*)((uintptr_t)current_ptr - vaddr); }",
        "{ page_table_store(dynarmic->page_table, idx, (void*)((uintptr_t)current_ptr - vaddr)); }", 2)
replace(cpp, "    size_t size = (1ULL << (PAGE_TABLE_ADDRESS_SPACE_BITS - DYN_PAGE_BITS)) * sizeof(void *);\n#ifdef MAP_NORESERVE",
        "    size_t size = (1ULL << (PAGE_TABLE_ADDRESS_SPACE_BITS - DYN_PAGE_BITS)) * sizeof(void *);\n"
        "#if defined(__PROSPERO__)\n    return ps5_page_table_create(size);\n#endif\n#ifdef MAP_NORESERVE")

boc = dyn / "backend/x64/block_of_code.cpp"
replace(boc, "    bool useProtect() const override { return false; }\n#else\n    static constexpr size_t DYNARMIC_PAGE_SIZE = 4096;",
        "    bool useProtect() const override { return false; }\n"
        "#elif defined(__PROSPERO__)\n"
        "    uint8_t* alloc(size_t size) override {\n"
        "        void* p = ps5_exec_allocate(size, reinterpret_cast<uintptr_t>(&ps5_exec_allocate));\n"
        "        if (p == nullptr) {\n"
        "            throw Xbyak::Error(Xbyak::ERR_CANT_ALLOC);\n"
        "        }\n"
        "        return static_cast<uint8_t*>(p);\n"
        "    }\n\n"
        "    void free(uint8_t* p) override {\n"
        "        ps5_exec_release(p);\n"
        "    }\n\n"
        "    bool useProtect() const override { return false; }\n"
        "#else\n    static constexpr size_t DYNARMIC_PAGE_SIZE = 4096;")
insert_before(boc, "class CustomXbyakAllocator", "#if defined(__PROSPERO__)\n#    include <ps5platform/exec.h>\n#endif\n\n")

eh = dyn / "backend/exception_handler_posix.cpp"
replace(eh, "void SigHandler::SigAction(int sig, siginfo_t* info, void* raw_context) {\n    ASSERT(sig == SIGSEGV || sig == SIGBUS);",
        "void SigHandler::SigAction(int sig, siginfo_t* info, void* raw_context) {\n    ASSERT(sig == SIGSEGV || sig == SIGBUS);\n"
        "#if defined(__PROSPERO__)\n"
        "    if (dynarmic_ps5_page_table_fault(reinterpret_cast<uintptr_t>(info->si_addr))) {\n"
        "        return;\n"
        "    }\n"
        "#endif")
insert_before(eh, "namespace Dynarmic::Backend",
              '#if defined(__PROSPERO__)\nextern "C" bool dynarmic_ps5_page_table_fault(uintptr_t addr);\n#endif\n\n')
header = dyn / "dynarmic.h"
replace(header, "KHASH_MAP_INIT_INT64(memory, t_memory_page)\n",
        "#define dynarmic_page_hash(key) (khint32_t)((((key) >> 12) * 0x9E3779B97F4A7C15ULL) >> 32)\n"
        "KHASH_INIT(memory, khint64_t, t_memory_page, 1, dynarmic_page_hash, kh_int64_hash_equal)\n")

lift = dyn / "externals/mcl/include/mcl/mp/typelist/lift_sequence.hpp"
replace(lift, "#include <type_traits>\n", "#include <type_traits>\n#include <utility>\n")
replace(lift, "    using type = list<std::integral_constant<T, values>...>;\n};\n",
        "    using type = list<std::integral_constant<T, values>...>;\n};\n\n"
        "template<class T, T... values>\n"
        "struct lift_sequence_impl<std::integer_sequence<T, values...>> {\n"
        "    using type = list<std::integral_constant<T, values>...>;\n"
        "};\n")
print("patched", root)
