import abc

from .dataset_reference import FileInputDatasetReference, FileOutputDatasetReference


class FileDatasetIOMixin:
    @property
    @abc.abstractmethod
    def dataset_reference(
        self,
    ) -> FileInputDatasetReference | FileOutputDatasetReference:
        ...

    @property
    def data_path(self) -> str:
        return self.dataset_reference.path
