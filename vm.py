#!/usr/bin/env python3
import json
from os.path import expanduser
from os import makedirs, rmdir, sysconf, environ, cpu_count
from sys import exit, stderr
from argparse import ArgumentParser
from pathlib import Path
import re
import subprocess
import math
from typing import List, Dict, Any
from dataclasses import dataclass, field

QEMU_ARCH = {
    "aarch64",
    "alpha",
    "arm",
    "cris",
    "hppa",
    "i386",
    "lm32",
    "m68k",
    "microblaze",
    "microblazeel",
    "mips",
    "mips64",
    "mips64el",
    "mipsel",
    "moxie",
    "nios2",
    "or1k",
    "ppc",
    "ppc64",
    "riscv32",
    "riscv64",
    "s390x",
    "sh4",
    "sh4eb",
    "sparc",
    "sparc64",
    "tricore",
    "unicore32",
    "x86_64",
    "xtensa",
    "xtensaeb",
}
VGA_OPT = {"cirrus", "std", "vmware", "qxl"}
BYTES_RE = re.compile("^[0-9]+[MG]$")
OPTION_FILE_NAME = "options.json"


def main():
    parser = ArgumentParser()
    parser.set_defaults(func=print_help(parser))

    sub_parsers = parser.add_subparsers()

    def add_subcommand(func, **kwargs):
        ret = sub_parsers.add_parser(func.__name__, **kwargs)
        ret.set_defaults(func=func)
        return ret

    add_subcommand(new, help="Create new vm")

    add_subcommand(ls, help="List all vms")

    run_parser = add_subcommand(run, help="Run vm")
    run_parser.add_argument("VM", help="vm to run")
    run_parser.add_argument(
        "-m", "--mount", action="append", help="Temporarely mount file")

    edit_parser = add_subcommand(edit, help="Edit vm options")
    edit_parser.add_argument("VM", help="vm name")

    args = parser.parse_args()
    args.func(args)


def print_help(parser: ArgumentParser):
    def ret(_):
        parser.print_help()
        exit(1)

    return ret


def new(args):
    cfg = load_config()
    vms = get_vms(cfg)
    name = ask(
        "VM name?",
        validator=lambda name: name not in vms,
        validator_msg="Name already taken",
    )
    arch = ask("Arch?", available=QEMU_ARCH, default="x86_64")
    backing = ask(
        "Backing store?", available=["raw", "qcow2"], default="qcow2")
    size = ask("Image size?", regex=BYTES_RE)
    memory = ask(
        "Memory size?",
        regex=BYTES_RE,
        default=str(mem_gb_round() // 4) + "G",
    )
    cores = int(
        ask(
            "Number of cores to give to vm?",
            regex=re.compile("^[1-9][0-9]*$"),
            default=max(1,
                        cpu_count() // 4),
        ))
    vm_dir = Path(cfg.vm_dir) / name
    makedirs(vm_dir)

    try:
        if cfg.disable_cow:
            subprocess.run(["chattr", "+C", vm_dir], check=True)
    except subprocess.CalledProcessError:
        rmdir(vm_dir)
        exit(f"Couldn't disable cow on {vm_dir}")

    try:
        image_file = (vm_dir / "img").with_suffix(f".{backing}")
        subprocess.run(["qemu-img", "create", image_file, "-f", backing, size],
                       check=True)
        with open(vm_dir / OPTION_FILE_NAME, "w") as fh:
            json.dump(
                Options(
                    mounted=[],
                    memory=memory,
                    arch=arch,
                    cores=cores,
                    vga="std",
                    raw_args=[],
                ).__dict__,
                fh,
                indent=4,
            )
            fh.write("\n")

    except subprocess.CalledProcessError:
        eprint("qemu-img failed")
        rmdir(vm_dir)
        exit(1)
    except FileNotFoundError:
        exit(f"qemu-img not installed")


def ls(_args):
    cfg = load_config()
    vms = get_vms(cfg)
    for name, data in vms.items():
        print(f"{name} [{data.options.arch}]")


def edit(args):
    cfg = load_config()
    vms = get_vms(cfg)
    editor = get_editor()
    try:
        subprocess.run([editor, vms[args.VM].options_path])
    except KeyError:
        exit(f"vm {args.VM} doesn't exist")
    except FileNotFoundError:
        exit(f"Can't launch {editor}.\n\
Set the env var EDITOR to your favorite editor")


def run(args):
    cfg = load_config()
    vms = get_vms(cfg)
    try:
        vm = vms[args.VM]
    except KeyError:
        exit(f"vm {args.VM} doesn't exist")

    if args.mount is not None:
        vm.options.mounted += args.mount

    qemu_cmd = [
        f"qemu-system-{vm.options.arch}",
        "-m",
        vm.options.memory,
        "-smp",
        str(vm.options.cores),
        "-boot",
        "menu=on",
        "-vga",
        vm.options.vga,
    ]
    if cfg.enable_kvm:
        qemu_cmd.append("-enable-kvm")

    for mount in vm.options.mounted:
        qemu_cmd += ["-cdrom", mount]

    qemu_cmd += vm.options.raw_args

    qemu_cmd.append(vm.img_path)

    try:
        subprocess.run(qemu_cmd)
    except subprocess.CalledProcessError:
        exit(1)
    except FileNotFoundError:
        exit(f"{qemu_cmd[0]} for arch {vm.options.arch} not installed")


class Validator:
    def validate(self, other: Any) -> bool:
        pass

    def msg(self) -> str:
        pass


@dataclass
class Instance(Validator):
    t: Any

    def validate(self, other: Any) -> bool:
        return isinstance(other, self.t)

    def msg(self) -> str:
        return f"Is not a {self.t.__name__}"


@dataclass
class ListOf(Validator):
    t: Any

    def validate(self, other: Any) -> bool:
        return isinstance(other, list) and all(
            isinstance(elem, self.t) for elem in other)

    def msg(self) -> str:
        return f"Is not a list of {self.t.__name__}"


@dataclass
class JsonObj(Validator):
    schema: Dict[str, Validator]
    errs: List[str] = field(default_factory=list)

    def validate(self, other: Any) -> bool:
        self.errs.clear()
        ret = True
        for k, validator in self.schema.items():
            try:
                if not validator.validate(other[k]):
                    self.errs.append(f"Key {k}: {validator.msg()}")
                    ret = False
            except KeyError:
                self.errs.append(f"Missing key {k}")
                ret = False
        return ret

    def msg(self) -> str:
        return "\n".join(self.errs)


@dataclass
class OneOf(Validator):
    options: List[Any]

    def validate(self, other: Any) -> bool:
        return other in self.options

    def msg(self) -> str:
        return "Is not one of {}".format(", ".join(self.options))


@dataclass
class MatchesRe(Validator):
    rx: re.Pattern

    def validate(self, other: Any) -> bool:
        return isinstance(other, str) and self.rx.match(other)

    def msg(self) -> str:
        return f"Doesn't match regex {self.rx.pattern}"


class ValidationError(Exception):
    pass


def validate(schema: Validator, obj: Any):
    if not schema.validate(obj):
        raise ValidationError(f"Object doesn't match schema: {schema.msg()}")
    return obj


def get_editor() -> str:
    return environ.get("EDITOR", "vi")


def eprint(*args, **kwargs):
    print(*args, **kwargs, file=stderr)


@dataclass
class Options:
    mounted: List[str]
    memory: str
    arch: str
    cores: int
    vga: str
    raw_args: List[str]


OPTIONS_SCHEMA = JsonObj({
    "mounted": ListOf(str),
    "arch": OneOf(QEMU_ARCH),
    "memory": MatchesRe(BYTES_RE),
    "cores": Instance(int),
    "vga": OneOf(VGA_OPT),
    "raw_args": ListOf(str),
})


@dataclass
class VmInfo:
    options: Options
    img_path: Path
    options_path: Path


def ask(question,
        available=[],
        validator=lambda _: True,
        validator_msg="Invalid input",
        regex=None,
        default=None) -> str:
    q = f"{question}[{default}] " if default is not None else question + " "
    while True:
        try:
            inp = input(q)
        except EOFError:
            exit(f"Aborted")

        if len(inp) == 0 and default is not None:
            return default

        if len(available) != 0:
            if inp in available:
                return inp
            else:
                print(f"Invalid input: {inp}")
                print("Available options: {}".format(", ".join(available)))
                continue

        if regex is not None:
            if regex.match(inp):
                return inp
            else:
                print(f"Input doesn't match regex '{regex.pattern}'")
            continue
        if validator(inp):
            return inp
        else:
            print(validator_msg)


def mem_gb_round() -> float:
    return math.ceil(
        sysconf('SC_PAGE_SIZE') * sysconf('SC_PHYS_PAGES') / 1024**3)


@dataclass
class Config:
    vm_dir: Path
    enable_kvm: bool
    disable_cow: bool


def get_vms(cfg: Config) -> Dict[str, VmInfo]:
    ret = {}
    for vm in cfg.vm_dir.glob("*"):
        if not vm.is_dir():
            continue
        try:
            options_path = vm / OPTION_FILE_NAME
            with open(options_path) as fh:
                options = Options(**validate(OPTIONS_SCHEMA, json.load(fh)))
            ret[vm.name] = VmInfo(
                options=options,
                img_path=next(vm.glob("img.*")),
                options_path=options_path,
            )
        except FileNotFoundError:
            eprint(f"Missing {OPTION_FILE_NAME} in {vm}")
        except StopIteration:
            eprint(f"Missing image file in dir {vm}")
        except json.JSONDecodeError as e:
            eprint(f"Can't decode {vm}: {e}")
        except ValidationError as e:
            eprint(f"Malformed {OPTION_FILE_NAME} in {vm}: {e}")
    return ret


CONFIG_SCHEMA = JsonObj({
    "vm_dir": Instance(str),
    "enable_kvm": Instance(bool),
    "disable_cow": Instance(bool),
})


def load_config() -> Config:
    path = Path(environ.get("XDG_CONFIG_DIR",
                            expanduser("~/.config"))) / "vm_config.json"
    try:
        with open(path) as fh:
            js = validate(CONFIG_SCHEMA, json.load(fh))
            js["vm_dir"] = Path(js["vm_dir"]).expanduser()

            return Config(**js)
    except FileNotFoundError:
        makedirs(Path(path).parent, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({
                "vm_dir": "~/vm",
                "enable_kvm": True,
                "disable_cow": False,
            },
                      fh,
                      indent=4)
            fh.write("\n")
        exit(f"Wrote default config to {path}")
    except ValidationError as e:
        exit(f"Malformed config in {path}: {e}")
    except json.JSONDecodeError as e:
        exit(f"Can't decode config in {path}: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit("Aborted")
