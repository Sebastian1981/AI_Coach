import getpass
import io
import ssl
import zipfile
from pathlib import Path

import certifi
from garminconnect import Garmin
from requests.adapters import HTTPAdapter


class _GarminTLSAdapter(HTTPAdapter):
    """HTTPAdapter with ssl.create_default_context() for Python 3.12+ compatibility.

    Root-Cause: ssl_wrap_socket calls load_verify_locations(certifi) on the
    ssl.create_default_context() context (because cert_verify sets conn.ca_certs),
    which breaks the TLS handshake with Garmin servers.
    Fix: override cert_verify → never set ca_certs/ca_cert_dir.
    Our context already has system certs + certifi loaded.
    """

    def __init__(self, **kwargs):
        ctx = ssl.create_default_context()
        ctx.load_verify_locations(certifi.where())
        self._ssl_context = ctx
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(
            request, verify, cert
        )
        pool_kwargs["ssl_context"] = self._ssl_context
        return host_params, pool_kwargs

    def cert_verify(self, conn, url, verify, cert):
        if url.lower().startswith("https") and verify:
            conn.cert_reqs = "CERT_REQUIRED"
        else:
            conn.cert_reqs = "CERT_NONE"
        conn.ca_certs = None
        conn.ca_cert_dir = None
        if cert:
            if not isinstance(cert, str):
                conn.cert_file = cert[0]
                conn.key_file = cert[1]
            else:
                conn.cert_file = cert
                conn.key_file = None


class GarminClient:
    """Garmin Connect client with token caching and TLS fix for Python 3.12+."""

    def __init__(self, token_store: str | Path = "../data/.garmin_tokens"):
        self.token_store = Path(token_store)
        self._client: Garmin | None = None
        self._activities: list[dict] | None = None

    def login(self) -> None:
        """Login with cached token or interactive credentials."""
        token_files_exist = (
            self.token_store.exists() and any(self.token_store.iterdir())
            if self.token_store.exists()
            else False
        )

        if token_files_exist:
            self._client = Garmin()
        else:
            email = input("Garmin Connect E-Mail: ")
            password = getpass.getpass("Passwort: ")
            self._client = Garmin(email, password)

        self._apply_tls_fix()
        self._client.login(tokenstore=str(self.token_store))
        print("Login erfolgreich.")

    def _apply_tls_fix(self) -> None:
        adapter = _GarminTLSAdapter(pool_connections=20, pool_maxsize=20)
        self._client.client._api_session.mount("https://", adapter)
        self._client.client.cs.mount("https://", adapter)

    def get_activities(self, num: int = 100) -> list[dict]:
        """Fetch activity list from Garmin Connect."""
        return self._client.get_activities(0, num)

    def download_activities(
        self, activities_dir: str | Path, num: int = 100
    ) -> list[Path]:
        """Download FIT files; skip already existing ones.

        Caches the activity metadata in ``self.activities``.
        Returns a sorted list of all .fit paths in activities_dir.
        """
        activities_dir = Path(activities_dir)
        activities_dir.mkdir(parents=True, exist_ok=True)

        self._activities = self.get_activities(num)

        for act in self._activities:
            act_id = act["activityId"]
            act_name = act.get("activityName", str(act_id))
            fit_path = activities_dir / f"{act_id}.fit"

            if not fit_path.exists():
                zip_data = self._client.download_activity(
                    act_id, dl_fmt=self._client.ActivityDownloadFormat.ORIGINAL
                )
                with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                    fit_names = [n for n in zf.namelist() if n.endswith(".fit")]
                    if fit_names:
                        fit_path.write_bytes(zf.read(fit_names[0]))
                        print(f"  Heruntergeladen: {fit_path.name}  ({act_name})")
                    else:
                        print(f"  Keine .fit in ZIP für {act_id} ({act_name})")
            else:
                print(f"  Bereits vorhanden: {fit_path.name}")

        fit_files = sorted(activities_dir.glob("*.fit"))
        print(f"\nGesamt: {len(fit_files)} .fit Datei(en) in {activities_dir.resolve()}")
        return fit_files

    @property
    def activities(self) -> list[dict]:
        """Activity metadata cached from the last download_activities() call."""
        return self._activities if self._activities is not None else []

    @property
    def raw(self) -> Garmin:
        """Access the underlying Garmin instance for direct API calls."""
        return self._client
