terraform {
  required_providers {
    aws     = { source = "hashicorp/aws", version = "~> 5.0" }
    archive = { source = "hashicorp/archive", version = "~> 2.0" }
  }
}

provider "aws" {
  region = var.region
}

variable "region" { default = "us-east-1" }
variable "bucket_name" { default = "volguard-vraj-2026-tf" }
variable "model_key" { default = "model.json" }
variable "function_name" { default = "volguard-predict" }

resource "aws_s3_bucket" "data" {
  bucket        = var.bucket_name
  force_destroy = true
}

resource "aws_s3_object" "model" {
  bucket = aws_s3_bucket.data.id
  key    = var.model_key
  source = "${path.module}/model.json"
  etag   = filemd5("${path.module}/model.json")
}

resource "aws_s3_object" "prices" {
  bucket = aws_s3_bucket.data.id
  key    = "stock_prices.csv"
  source = "${path.module}/stock_prices.csv"
  etag   = filemd5("${path.module}/stock_prices.csv")
}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/lambda_function.py"
  output_path = "${path.module}/function.zip"
}

resource "aws_iam_role" "lambda_role" {
  name = "volguard-lambda-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "s3_read" {
  name = "volguard-s3-read"
  role = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject"]
      Resource = "${aws_s3_bucket.data.arn}/*"
    }]
  })
}

resource "aws_lambda_function" "predict" {
  function_name    = var.function_name
  runtime          = "python3.12"
  handler          = "lambda_function.handler"
  role             = aws_iam_role.lambda_role.arn
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 10
  memory_size      = 128

  environment {
    variables = {
      VOLGUARD_BUCKET    = aws_s3_bucket.data.id
      VOLGUARD_MODEL_KEY = var.model_key
    }
  }
}

resource "aws_lambda_function_url" "url" {
  function_name      = aws_lambda_function.predict.function_name
  authorization_type = "NONE"
}

resource "aws_lambda_permission" "url_public" {
  statement_id           = "public-url"
  action                 = "lambda:InvokeFunctionUrl"
  function_name          = aws_lambda_function.predict.function_name
  principal              = "*"
  function_url_auth_type = "NONE"
}

output "function_url" {
  value = aws_lambda_function_url.url.function_url
}

output "bucket_name" {
  value = aws_s3_bucket.data.id
}
