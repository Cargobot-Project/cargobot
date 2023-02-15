# Cargobot

İlk önce docker'ın bilgisayara inik olduğuna emin olun.

Sonra ilk önce docker image'ı buildlemeniz lazım onu da Dockerfile'ın olduğu directory'e gidip -yani bu directory'deyken- aşağıdaki commandi çalıştırın.

```
$ docker build -t cargobot_image .
```

Sonra shell script'i çalıştırmanız lazım.

```
sh cargobot_build.sh
```

Şu an docker containerın içine girmiş olmanız lazım. Virtual environment (venv) önceden aktive edilmiş şekilde giremiyorsunuz. O yüzden aşağıdaki kod ile venv'i aktive edeceğiz.

```
. venv/bin/activate
```

Şimdi istediğiniz .py dosyalarını çalıştırabilirsiniz.
