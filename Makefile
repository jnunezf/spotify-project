NAME=spotify-project-jnunezf
COMMIT_ID=$(shell git rev-parse HEAD)


build-ml-api-heroku:
	docker build --build-arg PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} -t registry.heroku.com/spotify-project-jnunezf/web:$(COMMIT_ID) .

push-ml-api-heroku:
	docker push registry.heroku.com/spotify-project-jnunezf/web:$(COMMIT_ID)
