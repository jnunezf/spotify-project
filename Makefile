NAME=spotify-project-jnunezf

build-ml-api-heroku:
	docker build --build-arg PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} -t registry.heroku.com/spotify-project-jnunezf/web .

push-ml-api-heroku:
	docker push registry.heroku.com/spotify-project-jnunezf/web:latest
