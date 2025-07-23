from typing import Callable, MutableMapping, Any, Type, Sequence

from jinja2 import Environment, PackageLoader, PrefixLoader, BaseLoader, ChoiceLoader, Template, Undefined, \
    BytecodeCache
from jinja2.defaults import BLOCK_START_STRING, BLOCK_END_STRING, VARIABLE_START_STRING, VARIABLE_END_STRING, \
    COMMENT_START_STRING, COMMENT_END_STRING, LINE_STATEMENT_PREFIX, LINE_COMMENT_PREFIX, TRIM_BLOCKS, LSTRIP_BLOCKS, \
    NEWLINE_SEQUENCE, KEEP_TRAILING_NEWLINE
from jinja2.ext import Extension
from typing_extensions import Literal


class TemplateLoader(BaseLoader):
    def __init__(self, package_name: str, default_lang: str = 'en'):
        en_loader = PackageLoader(package_name, package_path="templates/en")
        zh_loader = PackageLoader(package_name, package_path="templates/zh")
        self.default_lang = default_lang
        self.loader_map: dict[str, list[BaseLoader]] = {
            'en': [en_loader],
            'zh': [zh_loader],
            'cn': [zh_loader],
        }
        self._loader = self._build_jinja_loader(self.loader_map)

    @staticmethod
    def _build_jinja_loader(loader_map: dict[str, list[BaseLoader]]):
        choice_loaders = dict((key, ChoiceLoader(loaders)) for (key, loaders) in loader_map.items())
        return PrefixLoader(choice_loaders)

    def get_source(self, environment: "Environment", template: str) -> tuple[str, str | None, Callable[[], bool] | None]:
        return self._loader.get_source(environment, template)

    def list_templates(self) -> list[str]:
        return self._loader.list_templates()

    def load(self, environment: Environment, name: str, globals: MutableMapping[str, Any] | None = None) -> Template:
        return self._loader.load(environment, name, globals)

    def add_loaders(self, *args: BaseLoader, **kwargs: BaseLoader | list[BaseLoader]):
        if args:
            if self.default_lang not in kwargs:
                kwargs[self.default_lang] = list(args)
            elif isinstance(kwargs[self.default_lang], BaseLoader):
                kwargs[self.default_lang] = list(args) + [kwargs[self.default_lang]]
            elif isinstance(loader, list):
                kwargs[self.default_lang] = list(args) + kwargs[self.default_lang]
            else:
                kwargs[self.default_lang] = list(args)
        for k, loader in kwargs.items():
            if k not in self.loader_map:
                self.loader_map[k] = []
            if isinstance(loader, BaseLoader):
                loaders = [loader]
            elif isinstance(loader, list):
                loaders = loader
            else:
                loaders = []
            self.loader_map[k] = loaders + self.loader_map[k]
        self._loader = self._build_jinja_loader(self.loader_map)


class TemplateEnvironment(Environment):
    def __init__(
            self,
            package_name: str,
            default_lang: str | None = None,
            block_start_string: str = BLOCK_START_STRING,
            block_end_string: str = BLOCK_END_STRING,
            variable_start_string: str = VARIABLE_START_STRING,
            variable_end_string: str = VARIABLE_END_STRING,
            comment_start_string: str = COMMENT_START_STRING,
            comment_end_string: str = COMMENT_END_STRING,
            line_statement_prefix: str | None = LINE_STATEMENT_PREFIX,
            line_comment_prefix: str | None = LINE_COMMENT_PREFIX,
            trim_blocks: bool = TRIM_BLOCKS,
            lstrip_blocks: bool = LSTRIP_BLOCKS,
            newline_sequence: Literal['\n', '\r\n', '\r'] = NEWLINE_SEQUENCE,
            keep_trailing_newline: bool = KEEP_TRAILING_NEWLINE,
            extensions: Sequence[str | Type["Extension"]] = (),
            optimized: bool = True,
            undefined: Type[Undefined] = Undefined,
            finalize: Callable[..., Any] | None = None,
            autoescape: bool | Callable[[str | None], bool] = False,
            cache_size: int = 400,
            auto_reload: bool = True,
            bytecode_cache: BytecodeCache | None = None,
            enable_async: bool = False):
        self.loader = TemplateLoader(package_name, default_lang or 'en')
        super().__init__(
            block_start_string=block_start_string,
            block_end_string=block_end_string,
            variable_start_string=variable_start_string,
            variable_end_string=variable_end_string,
            comment_start_string=comment_start_string,
            comment_end_string=comment_end_string,
            line_statement_prefix=line_statement_prefix,
            line_comment_prefix=line_comment_prefix,
            trim_blocks=trim_blocks,
            lstrip_blocks=lstrip_blocks,
            newline_sequence=newline_sequence,
            keep_trailing_newline=keep_trailing_newline,
            extensions=extensions,
            optimized=optimized,
            undefined=undefined,
            finalize=finalize,
            autoescape=autoescape,
            loader=self.loader,
            cache_size=cache_size,
            auto_reload=auto_reload,
            bytecode_cache=bytecode_cache,
            enable_async=enable_async
        )

    def add_loaders(self, *args: BaseLoader, **kwargs: BaseLoader | list[BaseLoader]):
        self.loader.add_loaders(*args, **kwargs)

    def load_template(self, name: str, lang: str | None = None, globals: MutableMapping[str, Any] | None = None):
        default_lang = self.loader.default_lang
        lang_options = set(self.loader.loader_map.keys())

        # Build candidate languages list by priority
        candidate_langs: list[str] = []
        if lang:
            candidate_langs.append(lang)
        if default_lang not in candidate_langs:
            candidate_langs.append(default_lang)
        if 'en' not in candidate_langs:
            candidate_langs.append('en')
        # Add remaining available languages (excluding already added ones)
        for l in lang_options:
            if l not in candidate_langs:
                candidate_langs.append(l)

        # Generate template paths
        template_names = [f"{l}/{name}" for l in candidate_langs]
        return self.select_template(names=template_names, globals=globals)
