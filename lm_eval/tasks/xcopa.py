"""

Homepage:
"""
import typing

from lm_eval.api.task import PromptSourceTask


_CITATION = """
TODO: add
"""

class XCopaBase(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "xcopa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return self.dataset["validation"]

    def test_docs(self):
        return self.dataset["test"]

    def invalid_doc_for_prompt(self, doc) -> bool:
        # HACK: Some copa templates have conditionals that ignore documents
        # when the condition is not met, like `{if doc['question'] != \"cause\"}`.
        # This means the prompt will never produce an input and target.
        try:
            result = self.prompt_template.apply(doc)
            if result == ['']:
                return True
            else:
                return False
        except Exception:
            return True

class XCopaId(XCopaBase):
    DATASET_NAME = "id"

class XCopaIt(XCopaBase):
    DATASET_NAME = "it"

class XCopaSw(XCopaBase):
    DATASET_NAME = "sw"

class XCopaTa(XCopaBase):
    DATASET_NAME = "ta"

class XCopaVi(XCopaBase):
    DATASET_NAME = "vi"

class XCopaZh(XCopaBase):
    DATASET_NAME = "zh"

XCOPA_TASKS = [
    XCopaId,
    XCopaIt,
    XCopaSw,
    XCopaTa,
    XCopaVi,
    XCopaZh,
]

def construct_tasks() -> typing.Dict[str, XCopaBase]:
    """
    Returns a dictionary of tasks keyed by task name, for example:
        "xcopa/id": XCopaId
    will dispatch to the GEM WikiLingua Arabic class.
    """
    tasks = {}
    for task_class in XCOPA_TASKS:
        benchmark = task_class.DATASET_PATH
        lang = task_class.DATASET_NAME
        tasks[f"{benchmark}_{lang}"] = task_class
    return tasks