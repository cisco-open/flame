const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api',
    createProxyMiddleware({
      target: 'https://flame-blue-b.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/api' : '/',
        },
    })
  );

  app.use(
    '/design/api',
    createProxyMiddleware({
      target: 'https://flame-blue-b.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/design/api' : '/',
        },
    })
  )

  app.use(
    '/jobs/api',
    createProxyMiddleware({
      target: 'https://flame-blue-b.int.dev.eticloud.io',
      changeOrigin: true,
      pathRewrite: {
        '^/jobs/api' : '/',
        },
    })
  )

  app.use(
    '/mlflow',
    createProxyMiddleware({
      target: 'https://flame-mlflow.int.dev.eticloud.io/ajax-api/2.0/mlflow',
      changeOrigin: true,
      pathRewrite: {
        '^/mlflow' : '/',
        },
    })
  )

  app.use(
    '/jobs/mlflow',
    createProxyMiddleware({
      target: 'https://flame-mlflow.int.dev.eticloud.io/ajax-api/2.0/mlflow',
      changeOrigin: true,
      pathRewrite: {
        '^/jobs/mlflow' : '/',
        },
    })
  )
};