"""Paketin hata sınıfları. Arayüz bu sınıfları kullanıcı metnine çevirir."""


class SkinLesionError(Exception):
    """Paketteki bütün beklenen hataların tabanı."""


class ConfigError(SkinLesionError):
    """Config dosyası eksik ya da geçersiz."""


class CheckpointNotFoundError(SkinLesionError):
    """Model ağırlıkları ya da checkpoint config'i bulunamadı."""


class InvalidImageError(SkinLesionError):
    """Girdi bir görüntü değil ya da kullanılamayacak kadar küçük."""


class TestSetAccessError(SkinLesionError):
    """Test kümesine `evaluate --final` dışında erişim denendi."""

    __test__ = False  # pytest bu sınıfı test sınıfı sanmasın.
