import boto3
import os


class S3PDFDownloader():

    def __init__(self, bucket_names):
        """Searching and downloading files in the specified buckets. 

        Args:
            bucket_names (List): 
                a list of bucket names to search
                the search orders is the same as the name orders in the list

        Examples:
            >>> downloader = S3PDFDownloader(['ai2-s2-pdfs', 'ai2-s2-pdfs-private'])
            >>> downloader.check_paper(test_sha)
            >>> res = downloader.download_paper(test_sha, 'papers/')
            >>> # res not False - Successful download returns the bucket_name
            >>> # res is False  - File not found on the s3 buckets 

        """
        self.bucket_names = bucket_names
        self.s3 = boto3.client('s3')

    @staticmethod
    def _construct_prefix(paper_sha):
        return f"{paper_sha[:4]}/{paper_sha[4:]}.pdf"

    def check_paper(self, paper_sha):
        """Check if a paper sha exists in the buckets. 

        Args:
            paper_sha (str): paper sha without `.pdf`

        Returns:
            [str or bool]: 
                returns the bucket name if found, 
                otherwise returns False.
        """
        for bucket_name in self.bucket_names:
            res = self.s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=self._construct_prefix(paper_sha),
            )
            if res['KeyCount'] > 0:
                return bucket_name

        return False

    def download_paper(self, paper_sha, save_path):
        """Download a paper from the buckets. 

        Args:
            paper_sha (str): paper sha without `.pdf`

        Returns:
            [str or bool]: 
                returns the bucket name if found
                and downloaded, otherwise returns False.
        """
        bucket_name = self.check_paper(paper_sha)

        if bucket_name:
            self.s3.download_file(bucket_name, self._construct_prefix(
                paper_sha), os.path.join(save_path, paper_sha+'.pdf'))

        return bucket_name
